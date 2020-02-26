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
package com.linkedin.photon.ml.algorithm

import org.apache.spark.sql.{DataFrame, SparkSession}

import com.linkedin.photon.ml.Types.{FeatureShardId, REType}
import com.linkedin.photon.ml.data.InputColumnsNames
import com.linkedin.photon.ml.function.ObjectiveFunction
import com.linkedin.photon.ml.function.ObjectiveFunctionHelper.{DistributedObjectiveFunctionFactory, ObjectiveFunctionFactoryFactory, SingleNodeObjectiveFunctionFactory}
import com.linkedin.photon.ml.model.Coefficients
import com.linkedin.photon.ml.normalization.NormalizationContext
import com.linkedin.photon.ml.optimization.DistributedOptimizationProblem
import com.linkedin.photon.ml.optimization.VarianceComputationType.VarianceComputationType
import com.linkedin.photon.ml.optimization.game.{CoordinateOptimizationConfiguration, FixedEffectOptimizationConfiguration, RandomEffectOptimizationConfiguration}
import com.linkedin.photon.ml.sampling.DownSampler
import com.linkedin.photon.ml.sampling.DownSamplerHelper.DownSamplerFactory
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import com.linkedin.photon.ml.util.PhotonBroadcast

/**
 * Factory to build [[Coordinate]] derived objects. Given generic input shared between coordinates, determine the type
 * of [[Coordinate]] to build, and do so.
 */
object CoordinateFactory {

  /**
   * Creates a [[Coordinate]] of the appropriate type, given the input data set,
   * [[CoordinateOptimizationConfiguration]], and [[ObjectiveFunction]].
   *
   * @param dataset The input data to use for training
   * @param featureShardId
   * @param inputColumnsNames
   * @param coordinateOptConfig The optimization settings for training
   * @param lossFunctionFactoryConstructor A constructor for the loss function factory function
   * @param glmConstructor A constructor for the type of [[GeneralizedLinearModel]] being trained
   * @param downSamplerFactory A factory function for the [[DownSampler]] (if down-sampling is enabled)
   * @param normalizationContext The [[NormalizationContext]]
   * @param varianceComputationType Should the trained coefficient variances be computed in addition to the means?
   * @param interceptIndexOpt The index of the intercept, if one is present
   * @param rETypeOpt
   * @return A [[Coordinate]] instance
   */
  def build(
      dataset: DataFrame,
      featureShardId: FeatureShardId,
      inputColumnsNames: InputColumnsNames,
      coordinateOptConfig: CoordinateOptimizationConfiguration,
      lossFunctionFactoryConstructor: ObjectiveFunctionFactoryFactory,
      glmConstructor: Coefficients => GeneralizedLinearModel,
      downSamplerFactory: DownSamplerFactory,
      normalizationContext: NormalizationContext,
      varianceComputationType: VarianceComputationType,
      interceptIndexOpt: Option[Int],
      rETypeOpt: Option[REType]): Coordinate = {

    val lossFunctionFactory = lossFunctionFactoryConstructor(coordinateOptConfig)

    (rETypeOpt, coordinateOptConfig, lossFunctionFactory) match {
      case (
          None,
          fEOptConfig: FixedEffectOptimizationConfiguration,
          distributedLossFunctionFactory: DistributedObjectiveFunctionFactory) =>

        val downSamplerOpt = if (DownSampler.isValidDownSamplingRate(fEOptConfig.downSamplingRate)) {
          Some(downSamplerFactory(fEOptConfig.downSamplingRate))
        } else {
          None
        }
        val normalizationPhotonBroadcast = PhotonBroadcast(
          SparkSession.builder.getOrCreate.sparkContext
            .broadcast(normalizationContext))

        new FixedEffectCoordinate(
          dataset,
          DistributedOptimizationProblem(
            fEOptConfig,
            distributedLossFunctionFactory(interceptIndexOpt),
            downSamplerOpt,
            glmConstructor,
            normalizationPhotonBroadcast,
            varianceComputationType),
          featureShardId,
          inputColumnsNames).asInstanceOf[Coordinate]

      case (
          Some(rEType),
          rEOptConfig: RandomEffectOptimizationConfiguration,
          singleNodeLossFunctionFactory: SingleNodeObjectiveFunctionFactory) =>

        RandomEffectCoordinate(
          dataset,
          rEType,
          featureShardId,
          inputColumnsNames,
          rEOptConfig,
          singleNodeLossFunctionFactory,
          glmConstructor,
          normalizationContext,
          varianceComputationType,
          interceptIndexOpt).asInstanceOf[Coordinate]

      case _ =>
        throw new UnsupportedOperationException(
          s"""Cannot build coordinate for the following input class combination:
          |  ${rETypeOpt.getOrElse("fixed-effect")}
          |  ${coordinateOptConfig.getClass.getName}
          |  ${lossFunctionFactory.getClass.getName}""".stripMargin)
    }
  }
}
