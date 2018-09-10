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
package com.linkedin.photon.ml.io

import com.linkedin.photon.ml.optimization.game._
import com.linkedin.photon.ml.data.{CoordinateDataConfiguration, FixedEffectDataConfiguration, RandomEffectDataConfiguration}
import com.linkedin.photon.ml.optimization._

/**
 * Trait for any class which defines a coordinate for coordinate descent.
 */
sealed trait CoordinateConfiguration {

  /**
   * Coordinate dataset definition
   */
  protected[ml] val dataConfiguration: CoordinateDataConfiguration

  /**
   * Coordinate optimization problem definition
   */
  protected[ml] val optimizationConfiguration: CoordinateOptimizationConfiguration

  /**
   * Regularization weights
   */
  protected[ml] val regularizationWeights: Set[Double]

  /**
   * Create a list of [[CoordinateOptimizationConfiguration]], one for each regularization weight, sorted in decreasing
   * order.
   *
   * @return A list of [[CoordinateOptimizationConfiguration]]
   */
  def expandOptimizationConfigurations: Seq[CoordinateOptimizationConfiguration]
}

/**
 * Definition of a fixed effect problem coordinate.
 *
 * @param dataConfiguration Coordinate dataset definition
 * @param optimizationConfiguration Coordinate optimization problem definition
 * @param regularizationWeights Regularization weights
 */
class FixedEffectCoordinateConfiguration private (
    override protected[ml] val dataConfiguration: FixedEffectDataConfiguration,
    override protected[ml] val optimizationConfiguration: FixedEffectOptimizationConfiguration,
    override protected[ml] val regularizationWeights: Set[Double])
  extends CoordinateConfiguration {

  require(regularizationWeights.nonEmpty, "At least one regularization weight required.")

  /**
   * Create a list of [[CoordinateOptimizationConfiguration]], one for each regularization weight, sorted in decreasing
   * order.
   *
   * @return A list of [[CoordinateOptimizationConfiguration]]
   */
  def expandOptimizationConfigurations: Seq[CoordinateOptimizationConfiguration] =
    regularizationWeights
      .toSeq
      .sorted
      .reverse
      .map { regWeight =>
        optimizationConfiguration.copy(regularizationWeight = regWeight)
      }
}

object FixedEffectCoordinateConfiguration {

  private val EMPTY_REG_WEIGHTS = Set(0D)

  /**
   * Helper function to generate a [[FixedEffectCoordinateConfiguration]].
   *
   * @param dataConfiguration Coordinate dataset definition
   * @param optimizationConfiguration Coordinate optimization problem definition
   * @param regularizationWeights Regularization weights
   * @return
   */
  def apply(
      dataConfiguration: FixedEffectDataConfiguration,
      optimizationConfiguration: FixedEffectOptimizationConfiguration,
      regularizationWeights: Set[Double] = EMPTY_REG_WEIGHTS): FixedEffectCoordinateConfiguration = {

    if (optimizationConfiguration.regularizationContext.regularizationType == RegularizationType.NONE) {
      new FixedEffectCoordinateConfiguration(dataConfiguration, optimizationConfiguration, EMPTY_REG_WEIGHTS)
    } else if (regularizationWeights == EMPTY_REG_WEIGHTS) {
      throw new IllegalArgumentException("Must provide regularization weight(s) if regularization enabled.")
    } else {
      new FixedEffectCoordinateConfiguration(dataConfiguration, optimizationConfiguration, regularizationWeights)
    }
  }
}

/**
 * Definition of a random effect problem coordinate.
 *
 * @param dataConfiguration Coordinate dataset definition
 * @param optimizationConfiguration Coordinate optimization problem definition
 * @param regularizationWeights Regularization weights
 */
class RandomEffectCoordinateConfiguration private (
    override protected[ml] val dataConfiguration: RandomEffectDataConfiguration,
    override protected[ml] val optimizationConfiguration: RandomEffectOptimizationConfiguration,
    override protected[ml] val regularizationWeights: Set[Double])
  extends CoordinateConfiguration {

  require(regularizationWeights.nonEmpty, "At least one regularization weight required.")

  /**
   * Create a list of [[CoordinateOptimizationConfiguration]], one for each regularization weight, sorted in decreasing
   * order.
   *
   * @return A list of [[CoordinateOptimizationConfiguration]]
   */
  def expandOptimizationConfigurations: Seq[CoordinateOptimizationConfiguration] =
    regularizationWeights
      .toSeq
      .sortBy(identity)
      .reverse
      .map { regWeight =>
        optimizationConfiguration.copy(regularizationWeight = regWeight)
      }
}

object RandomEffectCoordinateConfiguration {

  private val EMPTY_REG_WEIGHTS = Set(0D)

  /**
   * Helper function to generate a [[RandomEffectCoordinateConfiguration]].
   *
   * @param dataConfiguration Coordinate dataset definition
   * @param optimizationConfiguration Coordinate optimization problem definition
   * @param regularizationWeights Regularization weights
   * @return
   */
  def apply(
      dataConfiguration: RandomEffectDataConfiguration,
      optimizationConfiguration: RandomEffectOptimizationConfiguration,
      regularizationWeights: Set[Double] = EMPTY_REG_WEIGHTS): RandomEffectCoordinateConfiguration = {

    if (optimizationConfiguration.regularizationContext.regularizationType == RegularizationType.NONE) {
      new RandomEffectCoordinateConfiguration(dataConfiguration, optimizationConfiguration, EMPTY_REG_WEIGHTS)
    } else if (regularizationWeights == EMPTY_REG_WEIGHTS) {
      throw new IllegalArgumentException("Must provide regularization weight(s) if regularization enabled.")
    } else {
      new RandomEffectCoordinateConfiguration(dataConfiguration, optimizationConfiguration, regularizationWeights)
    }
  }
}
