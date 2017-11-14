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
   * Coordinate data set definition
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
 * @param dataConfiguration Coordinate data set definition
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
   * @param dataConfiguration Coordinate data set definition
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
 * @param dataConfiguration Coordinate data set definition
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
   * @param dataConfiguration Coordinate data set definition
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

/**
 * Definition of a factored random effect problem coordinate.
 *
 * @param dataConfiguration Coordinate data set definition
 * @param optimizationConfiguration Coordinate optimization problem definition
 * @param randomEffectRegularizationWeights Random effect regularization weights
 * @param latentEffectRegularizationWeights Latent factor regularization weights
 */
class FactoredRandomEffectCoordinateConfiguration private (
    override protected[ml] val dataConfiguration: RandomEffectDataConfiguration,
    override protected[ml] val optimizationConfiguration: FactoredRandomEffectOptimizationConfiguration,
    randomEffectRegularizationWeights: Set[Double],
    latentEffectRegularizationWeights: Set[Double])
  extends CoordinateConfiguration {

  require(randomEffectRegularizationWeights.nonEmpty, "At least one random effect regularization weight required.")
  require(latentEffectRegularizationWeights.nonEmpty, "At least one latent factor regularization weight required.")

  override protected[ml] val regularizationWeights: Set[Double] = Set()

  /**
   * Create a list of [[CoordinateOptimizationConfiguration]], one for each regularization weight, sorted in decreasing
   * order.
   *
   * @return A list of [[CoordinateOptimizationConfiguration]]
   */
  def expandOptimizationConfigurations: Seq[CoordinateOptimizationConfiguration] = {

    val sortedREWeights = randomEffectRegularizationWeights.toSeq.sorted.reverse
    val sortedLFWeights = latentEffectRegularizationWeights.toSeq.sorted.reverse

    for(reRegWeight <- sortedREWeights; lfRegWeight <- sortedLFWeights) yield {
      optimizationConfiguration.copy(
        reOptConfig = optimizationConfiguration.reOptConfig.copy(regularizationWeight = reRegWeight),
        lfOptConfig = optimizationConfiguration.lfOptConfig.copy(regularizationWeight = lfRegWeight))
    }
  }
}

object FactoredRandomEffectCoordinateConfiguration {

  private val EMPTY_REG_WEIGHTS = Set(0D)

  /**
   * Helper function to generate a [[FactoredRandomEffectCoordinateConfiguration]].
   *
   * @param dataConfiguration Coordinate data set definition
   * @param optimizationConfiguration Coordinate optimization problem definition
   * @param randomEffectRegularizationWeights Random effect regularization weights
   * @param latentEffectRegularizationWeights Latent factor regularization weights
   * @return
   */
  def apply(
      dataConfiguration: RandomEffectDataConfiguration,
      optimizationConfiguration: FactoredRandomEffectOptimizationConfiguration,
      randomEffectRegularizationWeights: Set[Double] = EMPTY_REG_WEIGHTS,
      latentEffectRegularizationWeights: Set[Double] = EMPTY_REG_WEIGHTS): FactoredRandomEffectCoordinateConfiguration = {

    val reRegType = optimizationConfiguration.reOptConfig.regularizationContext.regularizationType
    val lfRegType = optimizationConfiguration.lfOptConfig.regularizationContext.regularizationType

    val (reWeights, lfWeights) = (reRegType, lfRegType) match {
      case (RegularizationType.NONE, RegularizationType.NONE) => (EMPTY_REG_WEIGHTS, EMPTY_REG_WEIGHTS)
      case (RegularizationType.NONE, _) => (EMPTY_REG_WEIGHTS, latentEffectRegularizationWeights)
      case (_, RegularizationType.NONE) => (randomEffectRegularizationWeights, EMPTY_REG_WEIGHTS)
      case (_, _) => (randomEffectRegularizationWeights, latentEffectRegularizationWeights)
    }

    new FactoredRandomEffectCoordinateConfiguration(dataConfiguration, optimizationConfiguration, reWeights, lfWeights)
  }
}
