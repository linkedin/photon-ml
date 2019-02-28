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
import com.linkedin.photon.ml.function.ObjectiveFunctionHelper.ObjectiveFunctionFactory
import com.linkedin.photon.ml.function.{DistributedObjectiveFunction, ObjectiveFunction, SingleNodeObjectiveFunction}
import com.linkedin.photon.ml.model.Coefficients
import com.linkedin.photon.ml.normalization.NormalizationContext
import com.linkedin.photon.ml.optimization.{DistributedOptimizationProblem, SingleNodeOptimizationProblem}
import com.linkedin.photon.ml.optimization.VarianceComputationType.VarianceComputationType
import com.linkedin.photon.ml.optimization.game.{CoordinateOptimizationConfiguration, FixedEffectOptimizationConfiguration, RandomEffectOptimizationConfiguration, RandomEffectOptimizationProblem}
import com.linkedin.photon.ml.projector.IdentityProjection
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
   * @param lossFunctionConstructor A constructor for the loss function used for training
   * @param glmConstructor A constructor for the type of [[GeneralizedLinearModel]] being trained
   * @param downSamplerFactory A factory function for the [[DownSampler]] (if down-sampling is enabled)
   * @param trackState Should the internal optimization states be recorded?
   * @param normalizationContext The [[NormalizationContext]]
   * @param varianceComputationType Should the trained coefficient variances be computed in addition to the means?
   * @return A [[Coordinate]] for the [[Dataset]] of type [[D]]
   */
  def build[D <: Dataset[D]](
      dataset: D,
      coordinateOptConfig: CoordinateOptimizationConfiguration,
      lossFunctionConstructor: ObjectiveFunctionFactory,
      glmConstructor: (Coefficients) => GeneralizedLinearModel,
      downSamplerFactory: DownSamplerFactory,
      normalizationContext: NormalizationContext,
      varianceComputationType: VarianceComputationType,
      trackState: Boolean): Coordinate[D] = {

    val lossFunction: ObjectiveFunction = lossFunctionConstructor(coordinateOptConfig)

    (dataset, coordinateOptConfig, lossFunction) match {
      case (
        fEDataset: FixedEffectDataset,
        fEOptConfig: FixedEffectOptimizationConfiguration,
        distributedLossFunction: DistributedObjectiveFunction) =>

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
            distributedLossFunction,
            downSamplerOpt,
            glmConstructor,
            normalizationPhotonBroadcast,
            varianceComputationType,
            trackState)).asInstanceOf[Coordinate[D]]

      case (
        rEDataset: RandomEffectDataset,
        rEOptConfig: RandomEffectOptimizationConfiguration,
        singleNodeLossFunction: SingleNodeObjectiveFunction) =>

        rEOptConfig.projectionType match {
          case IdentityProjection =>
            val normalizationPhotonBroadcast = PhotonBroadcast(rEDataset.sparkContext.broadcast(normalizationContext))

            new RandomEffectCoordinate(
              rEDataset,
              new RandomEffectOptimizationProblem(
                rEDataset
                  .activeData
                  .mapValues { _ =>
                    SingleNodeOptimizationProblem(
                      rEOptConfig,
                      singleNodeLossFunction,
                      glmConstructor,
                      normalizationPhotonBroadcast,
                      varianceComputationType,
                      trackState)
                  },
                glmConstructor,
                trackState)).asInstanceOf[Coordinate[D]]

          case _ =>
            ProjectedRandomEffectCoordinate(
              rEDataset,
              rEOptConfig,
              singleNodeLossFunction,
              glmConstructor,
              normalizationContext,
              varianceComputationType,
              trackState).asInstanceOf[Coordinate[D]]
        }

      case _ =>
        throw new UnsupportedOperationException(
          s"""Cannot build coordinate for the following input class combination:
          |  ${dataset.getClass.getName}
          |  ${coordinateOptConfig.getClass.getName}
          |  ${lossFunction.getClass.getName}""".stripMargin)
    }
  }
}
