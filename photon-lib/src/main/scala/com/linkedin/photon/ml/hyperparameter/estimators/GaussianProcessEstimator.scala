/*
 * Copyright 2017 LinkedIn Corp. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License"); you may
 * not use this file except in compliance with the License. You may obtain a
 * copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */
package com.linkedin.photon.ml.hyperparameter.estimators

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.mean

import com.linkedin.photon.ml.hyperparameter.SliceSampler
import com.linkedin.photon.ml.hyperparameter.estimators.kernels._

/**
 * Estimates a Gaussian Process regression model
 *
 * @see "Gaussian Processes for Machine Learning" (GPML), http://www.gaussianprocess.org/gpml/, Chapter 2
 *
 * @param kernel the covariance kernel
 * @param normalizeLabels if true, the estimator normalizes labels to a mean of zero before fitting
 * @param noisyTarget learn a target function with noise
 * @param predictionTransformation transformation function to apply for predictions
 * @param monteCarloNumBurnInSamples the number of samples to draw during the burn-in phase of kernel parameter
 *   estimation
 * @param monteCarloNumSamples the number of samples to draw for estimating kernel parameters
 */
class GaussianProcessEstimator(
    kernel: Kernel = new RBF,
    normalizeLabels: Boolean = false,
    noisyTarget: Boolean = false,
    predictionTransformation: Option[PredictionTransformation] = None,
    monteCarloNumBurnInSamples: Int = 100,
    monteCarloNumSamples: Int = 10,
    seed: Long = System.currentTimeMillis) {

  val defaultNoise = 1e-4

  /**
   * Produces a Gaussian Process regression model from the input features and labels
   *
   * @param x the observed features
   * @param y the observed labels
   * @return the estimated model
   */
  def fit(x: DenseMatrix[Double], y: DenseVector[Double]): GaussianProcessModel = {
    require(x.rows > 0 && x.cols > 0, "Empty input.")
    require(x.rows == y.length, "Training feature sets and label sets must have the same number of elements")

    // Normalize labels
    val (yTrain, yMean) = if (normalizeLabels) {
      val m = mean(y)
      (y - m, m)
    } else {
      (y, 0.0)
    }

    val kernels = estimateKernelParams(x, yTrain)
    new GaussianProcessModel(x, yTrain, yMean, kernels, predictionTransformation)
  }

  /**
   * Estimates kernel parameters by sampling from the kernel parameter likelihood function
   *
   * We assume a uniform prior over the kernel parameters $\theta$ and observed features $x$, therefore:
   *
   *   $l(\theta|x,y) = p(y|theta,x) \propto p(theta|x,y)$
   *
   * Since the slice sampling algorithm requires that the function be merely proportional to the target distribution,
   * sampling from this function is equivalent to sampling from p(\theta|x,y). These samples can then be used to compute
   * a Monte Carlo estimate of the response for a new query point $q'$ by integrating over values of $\theta$:
   *
   *   $\int r(x', \theta) p(\theta) d\theta$
   *
   * In this way we (approximately) marginalize over all $\theta$ and arrive at a more robust estimate than would be
   * produced by computing a maximum likelihood point estimate of the parameters.
   *
   * @param x the observed features
   * @param y the observed labels
   * @return a collection of covariance kernels corresponding to the sampled kernel parameters
   */
  protected[estimators] def estimateKernelParams(
      x: DenseMatrix[Double],
      y: DenseVector[Double]): List[Kernel] = {

    val initialTheta = kernel.getInitialKernel(x, y).getParams

    // Sampler burn-in. Since Markov chain samplers like slice sampler exhibit serial dependence between samples, the
    // first n samples are biased by the initial choice of parameter vector. Here we perform a "burn in" procedure to
    // mitigate this.
    val thetaAfterBurnIn = (0 until monteCarloNumBurnInSamples)
      .foldLeft(initialTheta) { (currTheta, _) =>
        sampleNext(currTheta, x, y)
      }

    // Now draw the actual samples from the distribution
    val (_, samples) = (0 until monteCarloNumSamples)
      .foldLeft((thetaAfterBurnIn, List.empty[DenseVector[Double]])) { case ((currTheta, ls), _) =>
        val nextTheta = sampleNext(currTheta, x, y)
        (nextTheta, ls :+ nextTheta)
      }

    samples.map(kernel.withParams(_))
  }

  /**
   * Samples the next theta, given the previous one
   *
   * @param theta the previous sample
   * @param x the observed features
   * @param y the observed labels
   * @return the next theta sample
   */
  protected[estimators] def sampleNext(
      theta: DenseVector[Double],
      x: DenseMatrix[Double],
      y: DenseVector[Double]): DenseVector[Double] = {

    // Log likelihood wrapper function for learning length scale (holds noise and amplitude constant)
    def lengthScaleLogp(amplitudeNoise: DenseVector[Double]) =
      (ls: DenseVector[Double]) =>
        kernel
          .withParams(DenseVector.vertcat(amplitudeNoise, ls))
          .logLikelihood(x, y)

    // Log likelihood wrapper function for learning amplitude (holds noise and length scale constant)
    def amplitudeLogp(ls: DenseVector[Double]) =
      (amplitude: DenseVector[Double]) =>
        kernel
          .withParams(DenseVector.vertcat(amplitude, DenseVector(defaultNoise), ls))
          .logLikelihood(x, y)

    // Log likelihood wrapper function for learning amplitude and noise (holds length scale constant)
    def amplitudeNoiseLogp(ls: DenseVector[Double]) =
      (amplitudeNoise: DenseVector[Double]) =>
        kernel
          .withParams(DenseVector.vertcat(amplitudeNoise, ls))
          .logLikelihood(x, y)

    // Separate amplitude / noise, and length scale into separate vectors so that they can be sampled
    // separately. There is some interplay between these parameters, so the algorithm is a bit more well behaved if
    // they're sampled separately.
    val currAmplitudeNoise = theta.slice(0, 2)
    val currLengthScale = theta.slice(2, theta.length)
    val sampler = new SliceSampler(seed = seed)

    // Sample amplitude and noise
    val amplitudeNoise = if (noisyTarget) {
      sampler.draw(currAmplitudeNoise, amplitudeNoiseLogp(currLengthScale))

    } else {
      // If we're not sampling noise, just sample amplitude and concat on the default noise
      DenseVector.vertcat(
        sampler.draw(currAmplitudeNoise.slice(0, 1), amplitudeLogp(currLengthScale)),
        DenseVector(defaultNoise))
    }

    // Sample length scale
    val lengthScale = sampler.drawDimensionWise(currLengthScale, lengthScaleLogp(amplitudeNoise))

    DenseVector.vertcat(amplitudeNoise, lengthScale)
  }

}
