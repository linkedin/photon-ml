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
package com.linkedin.photon.ml.projector.vector

import java.util.Random

import breeze.linalg.{DenseMatrix, Matrix, Vector}

import com.linkedin.photon.ml.constants.MathConst

/**
 * Project [[Vector]] objects between two spaces using a projection matrix.
 *
 * @note Ultimately, we require:
 *
 *       w^t^ * v = w'^t^ * v'
 *
 *       where:
 *
 *       w is the coefficient vector in the original space
 *       v is a feature vector in the original space
 *       w' is the coefficient vector in the projected space
 *       v' is feature vector v in the projected space
 *
 *       If P is the projection matrix, then:
 *
 *       P * v = v'
 *
 *       and thus:
 *
 *       w^t^ * v = w'^t^ * (P * v)
 *       w^t^ = w'^t^ * P
 *       w = P^t^ * w'
 *
 * @param matrix The projection matrix
 * @throws UnsupportedOperationException Only dense matrices are currently supported for projection; an exception is
 *                                       thrown if any other type of matrix is passed
 */
protected[ml] case class MatrixProjector(matrix: Matrix[Double]) extends VectorProjector {

  private val projectionMatrix: DenseMatrix[Double] = matrix match {
    case denseMatrix: DenseMatrix[Double] => denseMatrix
    case _ => throw new UnsupportedOperationException(s"Projection matrix of class ${matrix.getClass} is not supported")
  }

  val projectedSpaceDimension: Int = matrix.rows
  val originalSpaceDimension: Int = matrix.cols
  val projectedInterceptId: Int = projectedSpaceDimension - 1

  /**
   * Project features into the new space.
   *
   * @param features The features
   * @return Projected features
   */
  override def projectForward(features: Vector[Double]): Vector[Double] = projectionMatrix * features

  /**
   * Project coefficients into the new space.
   *
   * @param coefficients The coefficients
   * @return Projected coefficients
   */
  override def projectBackward(coefficients: Vector[Double]): Vector[Double] = projectionMatrix.t * coefficients
}

object MatrixProjector {

  /**
   * Create a randomly generated matrix where components are drawn from the normal distribution
   * N(0, 1/projectedSpaceDimension).
   *
   * @param originalSpaceDimension The dimension of the original space, and within our context, this equals to the
   *                               number of columns of the random projection matrix
   * @param projectedSpaceDimension The dimension of the projected space, and within our context, this equals to the
   *                                number of rows of the random projection matrix
   * @param isKeepingInterceptTerm Whether to keep the intercept term in the original feature vector, which will be
   *                               done by adding a dummy row to the random projection matrix with all 0 but the last
   *                               element set to 1
   * @param seed The seed of random number generator
   * @return The dense Gaussian random projection matrix
   */
  protected[ml] def buildGaussianRandomMatrixProjector(
      originalSpaceDimension: Int,
      projectedSpaceDimension: Int,
      isKeepingInterceptTerm: Boolean,
      seed: Long = MathConst.RANDOM_SEED): MatrixProjector = {

    val random = new Random(seed)

    // A more conventional way to construct the Gaussian random projection matrix is to set
    // std = math.sqrt(projectedSpaceDimension), here we wish the magnitude of the matrix entries to be smaller, thus
    // adopt the following way
    val std = projectedSpaceDimension

    val matrix = if (isKeepingInterceptTerm) {
      Matrix.tabulate[Double](projectedSpaceDimension + 1, originalSpaceDimension) { (row, col) =>
        if (row < projectedSpaceDimension) {
          val rv = random.nextGaussian() / std

          if (rv > 1.0) {
            1.0
          } else if (rv < -1.0) {
            -1.0
          } else {
            rv
          }

        } else {
          if (col == originalSpaceDimension - 1) {
            1.0
          } else {
            0.0
          }
        }
      }

    } else {
      Matrix.tabulate[Double](projectedSpaceDimension, originalSpaceDimension)( (_, _) => random.nextGaussian() / std)
    }

    new MatrixProjector(matrix)
  }
}
