/*
 * Copyright 2018 LinkedIn Corp. All rights reserved.
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
package com.linkedin.photon.ml.optimization

import breeze.linalg.{DenseMatrix, Vector}

import com.linkedin.photon.ml.data.LabeledPoint

/**
 * Helper constants and functions used by both [[DistributedOptimizationProblemIntegTest]] and
 * [[SingleNodeOptimizationProblemIntegTest]].
 */
object OptimizationProblemIntegTestUtils {

  protected[optimization] val DATA_RANDOM_SEED: Int = 7
  protected[optimization] val WEIGHT_RANDOM_SEED = 100
  protected[optimization] val WEIGHT_RANDOM_MAX = 10
  protected[optimization] val DIMENSIONS: Int = 25
  protected[optimization] val TRAINING_SAMPLES: Int = DIMENSIONS * DIMENSIONS

  /**
   * Point-wise d^2^l/dz^2^ computation function for linear regression.
   *
   * @param coefficients Coefficient means vector
   * @param datum The next data point to process
   * @return The second derivative of the loss, with respect to the margin
   */
  def linearDzzLoss(coefficients: Vector[Double])(datum: LabeledPoint): Double = 1D

  /**
   * Point-wise d^2^l/dz^2^ computation function for logistic regression.
   *
   * @param coefficients Coefficient means vector
   * @param datum The next data point to process
   * @return The second derivative of the loss, with respect to the margin
   */
  def logisticDzzLoss(coefficients: Vector[Double])(datum: LabeledPoint): Double = {

    // For logistic regression, the second derivative of the loss function (with regard to z = X_i * B) is:
    //    sigmoid(z) * (1 - sigmoid(z))
    def sigmoid(z: Double): Double = 1.0 / (1.0 + math.exp(-z))

    val z: Double = datum.computeMargin(coefficients)
    val sigmoidValue: Double = sigmoid(z)

    sigmoidValue * (1.0 - sigmoidValue)
  }

  /**
   * Point-wise d^2^l/dz^2^ computation function for Poisson regression.
   *
   * @param coefficients Coefficient means vector
   * @param datum The next data point to process
   * @return The second derivative of the loss, with respect to the margin
   */
  def poissonDzzLoss(coefficients: Vector[Double])(datum: LabeledPoint): Double =
    math.exp(datum.computeMargin(coefficients))

  /**
   * Point-wise Hessian diagonal computation function.
   *
   * @param DzzLossFunction Point-wise d^2^l/dz^2^ computation function
   * @param matrix Current matrix (full or diagonal only) prior to processing the next data point
   * @param datum The next data point to process
   * @return The updated Hessian diagonal
   */
  def hessianSum(DzzLossFunction: (LabeledPoint) => Double)(matrix: DenseMatrix[Double], datum: LabeledPoint): DenseMatrix[Double] = {

    // For linear regression, the second derivative of the loss function (with regard to z = X_i * B) is 1.
    val features: Vector[Double] = datum.features
    val weight: Double = datum.weight
    val DzzLoss: Double = DzzLossFunction(datum)
    val x: DenseMatrix[Double] = features.toDenseVector.toDenseMatrix
    val hessian: DenseMatrix[Double] = (x.t * x) *:* (weight * DzzLoss)

    matrix + hessian
  }
}
