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

import breeze.linalg.{DenseMatrix, DenseVector, cholesky, diag, sum}
import breeze.numerics.constants.Pi
import breeze.numerics.log
import breeze.stats.mean

import com.linkedin.photon.ml.hyperparameter.Linalg.choleskySolve
import com.linkedin.photon.ml.hyperparameter.SliceSampler
import com.linkedin.photon.ml.hyperparameter.estimators.kernels._

/**
 * Estimates a Gaussian Process regression model
 *
 * @see "Gaussian Processes for Machine Learning" (GPML), http://www.gaussianprocess.org/gpml/, Chapter 2
 *
 * @param kernel the covariance kernel
 * @param normalizeLabels if true, the estimator normalizes labels to a mean of zero before fitting
 * @param predictionTransformation transformation function to apply for predictions
 * @param monteCarloNumBurnInSamples the number of samples to draw during the burn-in phase of kernel parameter
 *   estimation
 * @param monteCarloNumSamples the number of samples to draw for estimating kernel parameters
 */
class GaussianProcessEstimator(
    kernel: Kernel = new RBF,
    normalizeLabels: Boolean = false,
    predictionTransformation: Option[PredictionTransformation] = None,
    monteCarloNumBurnInSamples: Int = 100,
    monteCarloNumSamples: Int = 100,
    seed: Long = System.currentTimeMillis) {

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

    val logp = (theta: DenseVector[Double]) => logLikelihood(x, y, theta)
    val sampler = new SliceSampler(logp, range = kernel.getParamBounds, seed = seed)

    // Sampler burn-in. Since Markov chain samplers like slice sampler exhibit serial dependence between samples, the
    // first n samples are biased by the initial choice of parameter vector. Here we perform a "burn in" procedure to
    // mitigate this.
    val init = (0 until monteCarloNumBurnInSamples)
      .foldLeft(kernel.expandDimensions(kernel.getParams, x.cols)) { (currX, _) =>
        sampler.draw(currX)
      }

    // Now draw the real samples from the distribution
    val (_, samples) = (0 until monteCarloNumSamples)
      .foldLeft((init, List.empty[DenseVector[Double]])) { case ((currX, ls), _) =>
        val x = sampler.draw(currX)
        (x, ls :+ x)
      }

    samples.map(kernel.withParams(_))
  }

  /**
   * Computes the log likelihood of the given kernel parameters
   *
   * @param x the observed features
   * @param y the observed labels
   * @param theta the kernel parameters
   * @return the log likelihood
   */
  protected[estimators] def logLikelihood(
      x: DenseMatrix[Double],
      y: DenseVector[Double],
      theta: DenseVector[Double]): Double = {

    // Clone the kernel with given params, and apply it to the input
    val kern = kernel.withParams(theta)
    val k = kern(x)

    // Compute log likelihood. See GPML Algorithm 2.1
    try {
      // Line 2
      // Since we know the kernel function produces symmetric and positive definite matrices, we can use the Cholesky
      // factorization to solve the system $kx = y$ faster than a general purpose solver (e.g. LU) could.
      val l = cholesky(k)

      // Line 3
      val alpha = choleskySolve(l, y)

      // GPML algorithm 2.1 Line 7, equation 2.30
      -0.5 * (y.t * alpha) - sum(log(diag(l))) - k.rows/2.0 * log(2*Pi)

    } catch {
      case e: Exception => Double.NegativeInfinity
    }
  }
}
