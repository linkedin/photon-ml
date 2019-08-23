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

import com.linkedin.photon.ml.data.{Dataset, FixedEffectDataset, RandomEffectDataset}
import com.linkedin.photon.ml.function.ObjectiveFunctionHelper.{DistributedObjectiveFunctionFactory, ObjectiveFunctionFactoryFactory, SingleNodeObjectiveFunctionFactory}
import com.linkedin.photon.ml.function.ObjectiveFunction
import com.linkedin.photon.ml.model.{Coefficients, DatumScoringModel, FixedEffectModel, RandomEffectModel}
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
   * Creates a [[Coordinate]] of the appropriate type, given the input [[Dataset]],
   * [[CoordinateOptimizationConfiguration]], and [[ObjectiveFunction]].
   *
   * @tparam D Some type of [[Dataset]]
   * @param dataset The input data to use for training
   * @param coordinateOptConfig The optimization settings for training
   * @param lossFunctionFactoryConstructor A constructor for the loss function used for training
   * @param glmConstructor A constructor for the type of [[GeneralizedLinearModel]] being trained
   * @param downSamplerFactory A factory function for the [[DownSampler]] (if down-sampling is enabled)
   * @param normalizationContext The [[NormalizationContext]]
   * @param varianceComputationType Should the trained coefficient variances be computed in addition to the means?
   * @param priorModelOpt The prior model for warm-start and incremental training
   * @return A [[Coordinate]] for the [[Dataset]] of type [[D]]
   */
  def build[D <: Dataset[D]](
      dataset: D,
      coordinateOptConfig: CoordinateOptimizationConfiguration,
      lossFunctionFactoryConstructor: ObjectiveFunctionFactoryFactory,
      glmConstructor: Coefficients => GeneralizedLinearModel,
      downSamplerFactory: DownSamplerFactory,
      normalizationContext: NormalizationContext,
      varianceComputationType: VarianceComputationType,
      priorModelOpt: Option[DatumScoringModel],
      isIncrementalTrainingEnabled: Boolean = false): Coordinate[D] = {

    val lossFunctionFactory = lossFunctionFactoryConstructor(coordinateOptConfig, isIncrementalTrainingEnabled)

    (dataset, coordinateOptConfig, lossFunctionFactory, priorModelOpt) match {
      case (
        fEDataset: FixedEffectDataset,
        fEOptConfig: FixedEffectOptimizationConfiguration,
        distributedLossFunctionFactory: DistributedObjectiveFunctionFactory,
        fixedEffectModelOpt: Option[FixedEffectModel]) =>

        val downSamplerOpt = if (DownSampler.isValidDownSamplingRate(fEOptConfig.downSamplingRate)) {
          Some(downSamplerFactory(fEOptConfig.downSamplingRate))
        } else {
          None
        }
        val normalizationPhotonBroadcast = PhotonBroadcast(fEDataset.sparkContext.broadcast(normalizationContext))

        new FixedEffectCoordinate(
          fEDataset,
          DistributedOptimizationProblem(
            fEOptConfig,
            distributedLossFunctionFactory(fixedEffectModelOpt.map(_.model)),
            downSamplerOpt,
            glmConstructor,
            normalizationPhotonBroadcast,
            varianceComputationType)).asInstanceOf[Coordinate[D]]

      case (
        rEDataset: RandomEffectDataset,
        rEOptConfig: RandomEffectOptimizationConfiguration,
        singleLossFunctionFactory: SingleNodeObjectiveFunctionFactory,
        randomEffectModelOpt: Option[RandomEffectModel]) =>

        RandomEffectCoordinate(
          rEDataset,
          rEOptConfig,
          singleLossFunctionFactory,
          randomEffectModelOpt,
          glmConstructor,
          normalizationContext,
          varianceComputationType).asInstanceOf[Coordinate[D]]

      case _ =>
        throw new UnsupportedOperationException(
          s"""Cannot build coordinate for the following input class combination:
             |  ${dataset.getClass.getName}
             |  ${coordinateOptConfig.getClass.getName}
             |  ${lossFunctionFactory.getClass.getName}
             |  ${priorModelOpt.getClass.getName}""".stripMargin)
    }
  }
}
