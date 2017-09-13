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
protected[ml] sealed trait CoordinateConfiguration {

  /**
   * Coordinate data set definition
   */
  val dataConfiguration: CoordinateDataConfiguration

  /**
   * Coordinate optimization problem definition
   */
  val optimizationConfiguration: CoordinateOptimizationConfiguration

  /**
   * Regularization weights
   */
  val regularizationWeights: Set[Double]

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
protected[ml] class FixedEffectCoordinateConfiguration private (
    override val dataConfiguration: FixedEffectDataConfiguration,
    override val optimizationConfiguration: FixedEffectOptimizationConfiguration,
    override val regularizationWeights: Set[Double])
  extends CoordinateConfiguration {

  /**
   * Create a list of [[CoordinateOptimizationConfiguration]], one for each regularization weight, sorted in decreasing
   * order.
   *
   * @return A list of [[CoordinateOptimizationConfiguration]]
   */
  def expandOptimizationConfigurations: Seq[CoordinateOptimizationConfiguration] = {

    if (regularizationWeights.isEmpty) {
      Seq(optimizationConfiguration)
    } else {
      regularizationWeights
        .toSeq
        .sortBy(identity)
        .reverse
        .map { regWeight =>
          optimizationConfiguration.copy(regularizationWeight = regWeight)
        }
    }
  }
}

object FixedEffectCoordinateConfiguration {

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
      regularizationWeights: Set[Double] = Set()): FixedEffectCoordinateConfiguration = {

    if (optimizationConfiguration.regularizationContext.regularizationType == RegularizationType.NONE) {
      new FixedEffectCoordinateConfiguration(dataConfiguration, optimizationConfiguration, Set())
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
protected[ml] class RandomEffectCoordinateConfiguration private (
    override val dataConfiguration: RandomEffectDataConfiguration,
    override val optimizationConfiguration: RandomEffectOptimizationConfiguration,
    override val regularizationWeights: Set[Double])
  extends CoordinateConfiguration {

  /**
   * Create a list of [[CoordinateOptimizationConfiguration]], one for each regularization weight, sorted in decreasing
   * order.
   *
   * @return A list of [[CoordinateOptimizationConfiguration]]
   */
  def expandOptimizationConfigurations: Seq[CoordinateOptimizationConfiguration] = {

    if (regularizationWeights.isEmpty) {
      Seq(optimizationConfiguration)
    } else {
      regularizationWeights
        .toSeq
        .sortBy(identity)
        .reverse
        .map { regWeight =>
          optimizationConfiguration.copy(regularizationWeight = regWeight)
        }
    }
  }
}

object RandomEffectCoordinateConfiguration {

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
      regularizationWeights: Set[Double] = Set()): RandomEffectCoordinateConfiguration = {

    if (optimizationConfiguration.regularizationContext.regularizationType == RegularizationType.NONE) {
      new RandomEffectCoordinateConfiguration(dataConfiguration, optimizationConfiguration, Set())
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
protected[ml] class FactoredRandomEffectCoordinateConfiguration private (
    override val dataConfiguration: RandomEffectDataConfiguration,
    override val optimizationConfiguration: FactoredRandomEffectOptimizationConfiguration,
    randomEffectRegularizationWeights: Set[Double] = Set(),
    latentEffectRegularizationWeights: Set[Double] = Set())
  extends CoordinateConfiguration {

  override val regularizationWeights: Set[Double] = Set()

  /**
   * Create a list of [[CoordinateOptimizationConfiguration]], one for each regularization weight, sorted in decreasing
   * order.
   *
   * @return A list of [[CoordinateOptimizationConfiguration]]
   */
  def expandOptimizationConfigurations: Seq[CoordinateOptimizationConfiguration] =

    if (randomEffectRegularizationWeights.isEmpty && latentEffectRegularizationWeights.isEmpty) {
      Seq(optimizationConfiguration)

    } else if(latentEffectRegularizationWeights.isEmpty) {
      randomEffectRegularizationWeights
        .toSeq
        .sortBy(identity)
        .reverse
        .map { regWeight =>
          optimizationConfiguration.copy(
            reOptConfig = optimizationConfiguration.reOptConfig.copy(regularizationWeight = regWeight))
        }

    } else if(randomEffectRegularizationWeights.isEmpty) {
      latentEffectRegularizationWeights
        .toSeq
        .sortBy(identity)
        .reverse
        .map { regWeight =>
          optimizationConfiguration.copy(
            lfOptConfig = optimizationConfiguration.lfOptConfig.copy(regularizationWeight = regWeight))
        }

    } else {

      val sortedREWeights = randomEffectRegularizationWeights.toSeq.sortBy(identity).reverse
      val sortedLFWeights = randomEffectRegularizationWeights.toSeq.sortBy(identity).reverse

      for(reRegWeight <- sortedREWeights; lfRegWeight <- sortedLFWeights) yield {
        optimizationConfiguration.copy(
          reOptConfig = optimizationConfiguration.reOptConfig.copy(regularizationWeight = reRegWeight),
          lfOptConfig = optimizationConfiguration.lfOptConfig.copy(regularizationWeight = lfRegWeight))
      }
    }
}

object FactoredRandomEffectCoordinateConfiguration {

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
      randomEffectRegularizationWeights: Set[Double] = Set(),
      latentEffectRegularizationWeights: Set[Double] = Set()): FactoredRandomEffectCoordinateConfiguration = {

    val reRegType = optimizationConfiguration.reOptConfig.regularizationContext.regularizationType
    val lfRegType = optimizationConfiguration.lfOptConfig.regularizationContext.regularizationType

    val (reWeights, lfWeights) = (reRegType, lfRegType) match {
      case (RegularizationType.NONE, RegularizationType.NONE) => (Set[Double](), Set[Double]())
      case (RegularizationType.NONE, _) => (Set[Double](), latentEffectRegularizationWeights)
      case (_, RegularizationType.NONE) => (randomEffectRegularizationWeights, Set[Double]())
      case (_, _) => (randomEffectRegularizationWeights, latentEffectRegularizationWeights)
    }

    new FactoredRandomEffectCoordinateConfiguration(dataConfiguration, optimizationConfiguration, reWeights, lfWeights)
  }
}
