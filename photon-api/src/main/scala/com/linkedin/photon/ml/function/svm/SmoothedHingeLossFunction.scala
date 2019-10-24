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
package com.linkedin.photon.ml.function.svm

import breeze.linalg.Vector

import com.linkedin.photon.ml.algorithm.Coordinate
import com.linkedin.photon.ml.constants.MathConst
import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.function.ObjectiveFunction
import com.linkedin.photon.ml.optimization.game.{CoordinateOptimizationConfiguration, FixedEffectOptimizationConfiguration, RandomEffectOptimizationConfiguration}
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel

/**
 * Implement Rennie's smoothed hinge loss function (http://qwone.com/~jason/writing/smoothHinge.pdf) as an
 * optimizer-friendly approximation for linear SVMs. This Object is to the individual/distributed smoothed hinge loss
 * functions as the PointwiseLossFunction is to the individual/distributed GLM loss functions.
 *
 * @note Function names follow the differentiation notation found here:
 *       [[http://www.wikiwand.com/en/Notation_for_differentiation#/Euler.27s_notation]]
 */
object SmoothedHingeLossFunction {
  /**
   * Compute the loss and derivative of the smoothed hinge loss function at a single point.
   *
   * Note that the derivative is multiplied element-wise by the label in advance.
   *
   * @param margin The margin, i.e. z in l(z, y)
   * @param label The label, i.e. y in l(z, y)
   * @return The value and the 1st derivative
   */
  def lossAndDzLoss(margin: Double, label: Double): (Double, Double) = {

    val modifiedLabel = if (label < MathConst.POSITIVE_RESPONSE_THRESHOLD) -1D else 1D
    val z = modifiedLabel * margin

    // Eq: 2, page 2
    val loss = if (z <= 0) {
      0.5 - z
    } else if (z < 1) {
      0.5 * (1.0 - z) * (1.0 - z)
    } else {
      0.0
    }

    // Eq. 3, page 2
    val deriv = if (z < 0) {
      -1.0
    } else if (z < 1) {
      z - 1.0
    } else {
      0.0
    }

    (loss, deriv * modifiedLabel)
  }

  /**
   * Compute the loss and derivative of the smoothed hinge loss function at a single point.
   *
   * @param datum A single data point
   * @param coefficients The model coefficients
   * @param cumGradient The cumulative Gradient vector for all points in the dataset
   * @return The value at the given data point
   */
  def calculateAt(
    datum: LabeledPoint,
    coefficients: Vector[Double],
    cumGradient: Vector[Double]): Double = {

    val margin = datum.computeMargin(coefficients)
    val (loss, deriv) = lossAndDzLoss(margin, datum.label)

    // Eq. 5, page 2 (derivative multiplied by label in lossAndDerivative method)
    breeze.linalg.axpy(datum.weight * deriv, datum.features, cumGradient)
    datum.weight * loss
  }

  /**
   * Construct a factory function for building distributed and non-distributed smoothed hinge loss functions.
   *
   * @param treeAggregateDepth The tree-aggregate depth to use during aggregation
   * @param config Optimization problem configuration
   * @return A function which builds the appropriate type of [[ObjectiveFunction]] for a given [[Coordinate]] type and
   *         optimization settings.
   */
  def buildFactory(
      treeAggregateDepth: Int)(
      config: CoordinateOptimizationConfiguration): (Option[GeneralizedLinearModel], Option[Int]) => ObjectiveFunction =
    config match {
      case fEOptConfig: FixedEffectOptimizationConfiguration =>
        (_: Option[GeneralizedLinearModel], _: Option[Int]) =>
          DistributedSmoothedHingeLossFunction(fEOptConfig, treeAggregateDepth)

      case rEOptConfig: RandomEffectOptimizationConfiguration =>
        (_: Option[GeneralizedLinearModel], _: Option[Int]) =>
          SingleNodeSmoothedHingeLossFunction(rEOptConfig)

      case _ =>
        throw new UnsupportedOperationException(
          s"Cannot create a smoothed hinge loss linear SVM loss function from a coordinate configuration with class " +
            s"'${config.getClass.getName}'")
    }
}
