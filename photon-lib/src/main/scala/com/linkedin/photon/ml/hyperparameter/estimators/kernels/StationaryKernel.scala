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
package com.linkedin.photon.ml.hyperparameter.estimators.kernels

import breeze.linalg._
import breeze.numerics.constants.Pi
import breeze.numerics.{log, pow, sqrt}

import com.linkedin.photon.ml.util.Linalg.choleskySolve

/**
 * Base trait for stationary covariance kernel functions
 *
 * Stationary kernels depend on the relative positions of points (e.g. distance), rather than on their absolute
 * positions.
 *
 * @param amplitude the covariance amplitude
 * @param noise the observation noise
 * @param lengthScale the length scale of the kernel. This controls the complexity of the kernel, or the degree to which
 *   it can vary within a given region of the function's domain. Higher values allow less variation, and lower values
 *   allow more.
 * @param indexedTransformMap the map specifies the indices and transformation function of hyper-parameters
 */
abstract class StationaryKernel(
    amplitude: Double = 1.0,
    noise: Double = 1e-4,
    lengthScale: DenseVector[Double] = DenseVector(1.0),
    indexedTransformMap: Map[Int, Double => Double] = Map())
  extends Kernel {

  // Amplitude lognormal prior
  val amplitudeScale = 1.0

  // Noise horseshoe prior
  val noiseScale = 0.1

  // Length scale tophat prior
  val lengthScaleMax = 2.0

  /**
   * Computes the kernel function from the pairwise distances between points. Implementing classes should override this
   * to provide the specific kernel computation.
   *
   * @param dists the m x p matrix of pairwise distances between m and p points
   * @return the m x p covariance matrix
   */
  protected[kernels] def fromPairwiseDistances(dists: DenseMatrix[Double]): DenseMatrix[Double]

  /**
   * Unwraps hyper-parameters from scaled values to true values
   *
   * @param x the matrix of points, where each of the m rows is a point in the space and some columns contain scaled values
   * @return the unwrapped x that all the columns have true values
   */
  protected[kernels] def transform(x: DenseMatrix[Double]): DenseMatrix[Double] = {
    val xTransformed = x.copy
    indexedTransformMap.map { case (index: Int, rescalingFunction: (Double => Double)) =>
      xTransformed(::, index) := xTransformed(::, index).map(rescalingFunction)
    }
    xTransformed
  }

  /**
   * Applies the kernel function to the given points
   *
   * @param x the matrix of points, where each of the m rows is a point in the space
   * @return the m x m covariance matrix
   */
  override def apply(x: DenseMatrix[Double]): DenseMatrix[Double] = {
    require(x.rows > 0 && x.cols > 0, "Empty input.")

    val xTransformed = transform(x)
    val ls = expandDimensions(lengthScale, xTransformed.cols)
    val dists = pairwiseDistances(xTransformed(*,::) / ls)

    (amplitude * fromPairwiseDistances(dists)) +
      (noise * DenseMatrix.eye[Double](xTransformed.rows))
  }

  /**
   * Applies the kernel functions to the two sets of points
   *
   * @param x1 the matrix containing the first set of points, where each of the m rows is a point in the space
   * @param x2 the matrix containing the second set of points, where each of the p rows is a point in the space
   * @return the m x p covariance matrix
   */
  override def apply(x1: DenseMatrix[Double], x2: DenseMatrix[Double]): DenseMatrix[Double] = {
    require(x1.rows > 0 && x1.cols > 0 && x2.rows > 0, "Empty input.")
    require(x1.cols == x2.cols, "Inputs must have the same number of columns")

    val x1Transformed = transform(x1)
    val x2Transformed = transform(x2)
    val ls = expandDimensions(lengthScale, x1Transformed.cols)
    val dists = pairwiseDistances(x1Transformed(*,::) / ls, x2Transformed(*,::) / ls)

    amplitude * fromPairwiseDistances(dists)
  }

  /**
   * Returns the kernel parameters as a vector
   *
   * @return the kernel parameters
   */
  override def getParams: DenseVector[Double] = DenseVector.vertcat(DenseVector(amplitude, noise), lengthScale)

  /**
   * Computes the log likelihood of the kernel parameters
   *
   * @param x the observed features
   * @param y the observed labels
   * @return the log likelihood
   */
  override def logLikelihood(x: DenseMatrix[Double], y: DenseVector[Double]): Double = {
    // Bounds checks
    if (amplitude < 0.0 ||
        noise < 0.0 ||
        any(lengthScale :< 0.0)) {
      return Double.NegativeInfinity
    }

    // Tophat prior for length scale
    if (any(lengthScale :> lengthScaleMax)) {
      return Double.NegativeInfinity
    }

    // Apply the kernel to the input
    val k = apply(x)

    // Compute log likelihood. See GPML Algorithm 2.1
    try {
      // Line 2
      // Since we know the kernel function produces symmetric and positive definite matrices, we can use the Cholesky
      // factorization to solve the system $kx = y$ faster than a general purpose solver (e.g. LU) could.
      val l = cholesky(k)

      // Line 3
      val alpha = choleskySolve(l, y)

      // GPML algorithm 2.1 Line 7, equation 2.30
      val likelihood = -0.5 * (y.t * alpha) - sum(log(diag(l))) - k.rows/2.0 * log(2*Pi)

      // Add in lognormal prior for amplitude and horseshoe prior for noise
      likelihood +
        -0.5 * pow(log(sqrt(amplitude / amplitudeScale)), 2) +
        (if (noise > 0) {
          log(log(1.0 + pow(noiseScale / noise, 2)))
        } else {
          0
        })

    } catch {
      case e: Exception => Double.NegativeInfinity
    }
  }

  /**
   * Computes the pairwise squared distances between all points
   *
   * @param x the matrix of points, where each of the m rows is a point in the space
   * @return the m x m matrix of distances
   */
  protected[kernels] def pairwiseDistances(x: DenseMatrix[Double]): DenseMatrix[Double] = {
    val out = DenseMatrix.zeros[Double](x.rows, x.rows)

    for (i <- 0 until x.rows) {
      // Lower triangular first, then reflect it across the diagonal rather than recomputing
      for (j <- 0 until i) {
        val dist = squaredDistance(x(i, ::).t, x(j, ::).t)
        out(i, j) = dist
        out(j, i) = dist
      }
    }

    out
  }

  /**
   * Computes the pairwise squared distance between the points in two sets
   *
   * @param x1 the matrix containing the first set of points, where each of the m rows is a point in the space
   * @param x2 the matrix containing the second set of points, where each of the p rows is a point in the space
   * @return the m x p matrix of distances
   */
  protected[kernels] def pairwiseDistances(x1: DenseMatrix[Double], x2: DenseMatrix[Double]): DenseMatrix[Double] = {
    val out = DenseMatrix.zeros[Double](x1.rows, x2.rows)

    for (i <- 0 until x1.rows) {
      for (j <- 0 until x2.rows) {
        out(i, j) = squaredDistance(x1(i, ::).t, x2(j, ::).t)
      }
    }

    out
  }

}
