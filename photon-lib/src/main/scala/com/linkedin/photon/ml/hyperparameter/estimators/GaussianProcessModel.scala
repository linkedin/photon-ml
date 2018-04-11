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

import breeze.linalg.{DenseMatrix, DenseVector, cholesky, diag}

import com.linkedin.photon.ml.hyperparameter.estimators.kernels._
import com.linkedin.photon.ml.util.Linalg.{choleskySolve, vectorMean}

/**
 * Gaussian Process regression model that predicts mean and variance of response for new observations
 *
 * @see Gaussian Processes for Machine Learning (GPML), http://www.gaussianprocess.org/gpml/, Chapter 2
 *
 * @param xTrain the observed training features
 * @param yTrain the observed training labels
 * @param yMean the mean of the observed labels
 * @param kernels the sampled kernels
 * @param predictionTransformation optional transformation function that will be applied to the predicted response for
 *   each sampled kernel
 */
class GaussianProcessModel protected[estimators] (
    xTrain: DenseMatrix[Double],
    yTrain: DenseVector[Double],
    yMean: Double,
    kernels: Seq[Kernel],
    predictionTransformation: Option[PredictionTransformation]) {

  require(xTrain.rows > 0 && xTrain.cols > 0, "Empty training set.")
  require(xTrain.rows == yTrain.length, "Training feature sets and label sets must have the same number of elements")

  val featureDimension = xTrain.cols

  // Precompute items that don't depend on new data
  private val precomputedKernelVals = kernels.map { kernel =>
    val k = kernel(xTrain)

    // GPML Algorithm 2.1, Line 2
    // Since we know the kernel function produces symmetric and positive definite matrices, we can use the Cholesky
    // factorization to solve the system $kx = yTrain$ faster than a general purpose solver (e.g. LU) could
    val l = cholesky(k)

    // Line 3
    val alpha = choleskySolve(l, yTrain)

    kernel -> (l, alpha)
  }.toMap

  /**
   * Predicts mean and variance of response for new observations
   *
   * @param x the observed features
   * @return predicted mean and variance of response
   */
  def predict(x: DenseMatrix[Double]): (DenseVector[Double], DenseVector[Double]) = {
    require(x.rows > 0 && x.cols > 0, "Empty input.")
    require(x.cols == featureDimension, s"Model was trained for $featureDimension features, but input has ${x.cols}")

    val (means, vars) = kernels.map(predictWithKernel(x, _)).unzip
    (vectorMean(means), vectorMean(vars))
  }

  /**
   * Predicts and transforms the response for the new observations
   *
   * @param x the observed features
   * @return the transformed response prediction
   */
  def predictTransformed(x: DenseMatrix[Double]): DenseVector[Double] = {
    require(x.rows > 0 && x.cols > 0, "Empty input.")
    require(x.cols == featureDimension, s"Model was trained for $featureDimension features, but input has ${x.cols}")

    vectorMean(kernels
      .map(predictWithKernel(x, _))
      .map { case (means, covs) =>
        predictionTransformation.map(_(means, covs)).getOrElse(means)
      })
  }

  /**
   * Computes the predicted mean and variance of response for the new observation, given a single kernel
   *
   * @param x the observed features
   * @param kernel the covariance kernel
   * @return predicted mean and variance of response
   */
  protected[estimators] def predictWithKernel(
      x: DenseMatrix[Double],
      kernel: Kernel): (DenseVector[Double], DenseVector[Double]) = {

    val (l, alpha) = precomputedKernelVals(kernel)
    val ktrans = kernel(x, xTrain).t

    // GPML Algorithm 2.1, Line 4
    val yPred = ktrans.t * alpha

    // Line 5
    val v = l \ ktrans

    // Line 6
    val kx = kernel(x)
    val yCov = kx - v.t * v

    (yPred + yMean, diag(yCov))
  }
}
