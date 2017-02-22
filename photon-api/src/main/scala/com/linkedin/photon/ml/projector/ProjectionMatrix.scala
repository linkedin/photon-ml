/*
 * Copyright 2016 LinkedIn Corp. All rights reserved.
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
package com.linkedin.photon.ml.projector

import java.util.Random

import breeze.linalg.{DenseMatrix, Matrix, Vector, norm}
import breeze.stats.meanAndVariance

import com.linkedin.photon.ml.constants.MathConst
import com.linkedin.photon.ml.util.Summarizable

/**
 * A class for projection matrix that is used to project the features from their original space to a different
 * (usually, a lower dimensional) space.
 *
 * @param matrix The projection matrix. Currently, only dense matrices are supported for projection. An exception
 *               is thrown if any other type of matrix is passed
 */
protected[ml] case class ProjectionMatrix(matrix: Matrix[Double]) extends Projector with Summarizable {
  matrix match {
    case x: DenseMatrix[Double] =>
    case _ => throw new UnsupportedOperationException(s"Projection matrix of class ${matrix.getClass} for features " +
      s"projection operation is not supported")
  }

  override val projectedSpaceDimension = matrix.rows
  override val originalSpaceDimension = matrix.cols

  /**
   * Project features into the new space.
   *
   * @param features The features
   * @return Projected features
   */
  override def projectFeatures(features: Vector[Double]): Vector[Double] = {
    matrix * features
  }

  /**
   * Project coefficients into the new space.
   *
   * @param coefficients The coefficients
   * @return Projected coefficients
   */
  override def projectCoefficients(coefficients: Vector[Double]): Vector[Double] = {
    matrix match {
      case dm: DenseMatrix[Double] =>  dm.t * coefficients
      case _ => throw new RuntimeException("Should never reach here! Matrix should already be validated to be dense")
    }
  }

  /**
   *
   * @return A summary of the object in string representation
   */
  override def toSummaryString: String = {
    val stringBuilder = new StringBuilder()
    stringBuilder.append(s"meanAndVarianceAndCount of the flattened matrix: ${meanAndVariance(matrix.flatten())}")
    stringBuilder.append(s"\nmeanAndVarianceAndCount of the squared flattened matrix: " +
        s"${meanAndVariance(matrix.flatten().map(ele => ele * ele))}")
    stringBuilder.append(s"\nl2 norm of the flattened matrix: ${norm(matrix.flatten(), 2)}")
    stringBuilder.toString()
  }
}

object ProjectionMatrix {

  /**
   * Creating a randomly generated matrix where components are drawn from the normal distribution
   * N(0, 1/projectedSpaceDimension).
   *
   * @param projectedSpaceDimension The dimension of the projected space, and within our context, this equals to the
   *                                number of rows of the random projection matrix
   * @param originalSpaceDimension The dimension of the original space, and within our context, this equals to the
   *                               number of columns of the random projection matrix
   * @param isKeepingInterceptTerm Whether to keep the intercept term in the original feature vector, which will be
   *                               done by adding a dummy row to the random projection matrix with all 0 but the last
   *                               element set to 1
   * @param seed The seed of random number generator
   * @return The dense Gaussian random projection matrix
   */
  protected[ml] def buildGaussianRandomProjectionMatrix(
      projectedSpaceDimension: Int,
      originalSpaceDimension: Int,
      isKeepingInterceptTerm: Boolean,
      seed: Long = MathConst.RANDOM_SEED): ProjectionMatrix = {

    val random = new Random(seed)

    // A more conventional way to construct the Gaussian random projection matrix is to set
    // std = math.sqrt(projectedSpaceDimension), here we wish the magnitude of the matrix entries to be smaller, thus
    // adopt the following way
    val std = projectedSpaceDimension

    val matrix = if (isKeepingInterceptTerm) {
      Matrix.tabulate[Double](projectedSpaceDimension + 1, originalSpaceDimension){(row, col) =>
        if (row < projectedSpaceDimension) {
          val rv = random.nextGaussian() / std
          if (rv > 1.0) 1.0 else if (rv < -1.0) -1.0 else rv
        } else {
          if (col == originalSpaceDimension - 1) 1.0 else 0.0
        }
      }
    } else {
      Matrix.tabulate[Double](projectedSpaceDimension, originalSpaceDimension)((row, col) =>
        random.nextGaussian() / std
      )
    }

    new ProjectionMatrix(matrix)
  }
}
