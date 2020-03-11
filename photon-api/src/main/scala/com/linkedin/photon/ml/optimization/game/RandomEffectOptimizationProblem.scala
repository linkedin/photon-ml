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
package com.linkedin.photon.ml.optimization.game

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import com.linkedin.photon.ml.Types.REId
import com.linkedin.photon.ml.function.SingleNodeObjectiveFunction
import com.linkedin.photon.ml.model.{Coefficients, RandomEffectModel}
import com.linkedin.photon.ml.normalization.NormalizationContext
import com.linkedin.photon.ml.optimization.VarianceComputationType.VarianceComputationType
import com.linkedin.photon.ml.optimization.{SingleNodeOptimizationProblem, VarianceComputationType}
import com.linkedin.photon.ml.projector.LinearSubspaceProjector
import com.linkedin.photon.ml.spark.RDDLike
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import com.linkedin.photon.ml.util.PhotonNonBroadcast

/**
 * Representation for a random effect optimization problem.
 *
 * Q: Why shard the optimization problems?
 * A: In the future, we want to be able to have unique regularization weights per optimization problem. In addition, it
 *    may be useful to have access to the optimization state of each individual problem.
 *
 * @tparam Objective The objective function to optimize
 * @param optimizationProblems The component optimization problems (one per individual) for a random effect
 *                             optimization problem
 * @param glmConstructor The function to use for producing [[GeneralizedLinearModel]] objects from trained
 *                       [[Coefficients]]
 */
protected[ml] class RandomEffectOptimizationProblem[Objective <: SingleNodeObjectiveFunction](
    val optimizationProblems: RDD[(REId, SingleNodeOptimizationProblem[Objective])],
    glmConstructor: Coefficients => GeneralizedLinearModel)
  extends RDDLike {

  /**
   * Get the Spark context.
   *
   * @return The Spark context
   */
  override def sparkContext: SparkContext = optimizationProblems.sparkContext

  /**
   * Assign a given name to [[optimizationProblems]].
   *
   * @note Not used to reference models in the logic of photon-ml, only used for logging currently.
   *
   * @param name The parent name for all [[RDD]]s in this class
   * @return This object with the name of [[optimizationProblems]] assigned
   */
  override def setName(name: String): this.type = {

    optimizationProblems.setName(s"$name: Optimization problems")

    this
  }

  /**
   * Set the storage level of [[optimizationProblems]], and persist their values across the cluster the first time they
   * are computed.
   *
   * @param storageLevel The storage level
   * @return This object with the storage level of [[optimizationProblems]] set
   */
  override def persistRDD(storageLevel: StorageLevel): this.type = {

    if (!optimizationProblems.getStorageLevel.isValid) optimizationProblems.persist(storageLevel)

    this
  }

  /**
   * Mark [[optimizationProblems]] as non-persistent, and remove all blocks for them from memory and disk.
   *
   * @return This object with [[optimizationProblems]] marked non-persistent
   */
  override def unpersistRDD(): this.type = {

    if (optimizationProblems.getStorageLevel.isValid) optimizationProblems.unpersist()

    this
  }

  /**
   * Materialize [[optimizationProblems]] (Spark [[RDD]]s are lazy evaluated: this method forces them to be evaluated).
   *
   * @return This object with [[optimizationProblems]] materialized
   */
  override def materialize(): this.type = {

    optimizationProblems.count()

    this
  }

  /**
   * Create a default generalized linear model with 0-valued coefficients
   *
   * @param dimension The dimensionality of the model coefficients
   * @return A model with zero coefficients
   */
  def initializeModel(dimension: Int): GeneralizedLinearModel =
    glmConstructor(Coefficients.initializeZeroCoefficients(dimension))
}

object RandomEffectOptimizationProblem {

  /**
   * Build a new [[RandomEffectOptimizationProblem]] to optimize.
   *
   * @tparam RandomEffectObjective The type of objective function used to solve individual random effect optimization
   *                               problems
   * @param linearSubspaceProjectorsRDD The per-entity [[LinearSubspaceProjector]] objects used to compress the
   *                                    per-entity feature spaces
   * @param configuration The optimization problem configuration
   * @param objectiveFunctionFactory Factory for the objective function
   * @param glmConstructor The function to use for producing GLMs from trained coefficients
   * @param normalizationContext The normalization context
   * @param varianceComputationType If and how coefficient variances should be computed
   * @param interceptIndexOpt The option of intercept index
   * @return A new [[RandomEffectOptimizationProblem]]
   */
  protected[ml] def apply[RandomEffectObjective <: SingleNodeObjectiveFunction](
      linearSubspaceProjectorsRDD: RDD[(REId, LinearSubspaceProjector)],
      configuration: RandomEffectOptimizationConfiguration,
      objectiveFunctionFactory: (Option[GeneralizedLinearModel], Option[Int]) => RandomEffectObjective,
      priorRandomEffectModelOpt: Option[RandomEffectModel],
      glmConstructor: Coefficients => GeneralizedLinearModel,
      normalizationContext: NormalizationContext,
      varianceComputationType: VarianceComputationType = VarianceComputationType.NONE,
      interceptIndexOpt: Option[Int]): RandomEffectOptimizationProblem[RandomEffectObjective] = {

    val sc = linearSubspaceProjectorsRDD.sparkContext
    val configurationBroadcast = sc.broadcast(configuration)
    val objectiveFunctionBuilderBroadcast = sc.broadcast(objectiveFunctionFactory)
    val glmConstructorBroadcast = sc.broadcast(glmConstructor)
    val normalizationContextBroadcast = sc.broadcast(normalizationContext)

    // Generate new NormalizationContext and SingleNodeOptimizationProblem objects
    val optimizationProblems = linearSubspaceProjectorsRDD
      .leftOuterJoin(priorRandomEffectModelOpt.map(_.modelsRDD).getOrElse(sc.emptyRDD[(REId, GeneralizedLinearModel)]))
      .mapValues { case (projector: LinearSubspaceProjector, priorModelOpt: Option[GeneralizedLinearModel]) =>
        val normContext = normalizationContextBroadcast.value
        val factors = normContext.factorsOpt.map(factors => projector.projectForward(factors))
        val shiftsAndIntercept = normContext
          .shiftsAndInterceptOpt
          .map { case (shifts, intercept) =>
            val newShifts = projector.projectForward(shifts)
            val newIntercept = projector.originalToProjectedSpaceMap(intercept)

            (newShifts, newIntercept)
          }
        val projectedNormalizationContext = new NormalizationContext(factors, shiftsAndIntercept)
        val objectiveFunctionBuilder = objectiveFunctionBuilderBroadcast.value
        val projectedInterceptOpt = interceptIndexOpt.map { interceptIndex =>
          projector.originalToProjectedSpaceMap(interceptIndex)
        }

        // Project prior model coefficients
        val projectedPriorModelOpt = priorModelOpt.map{
          model =>
            val oldCoefficients = model.coefficients
            val newCoefficients = Coefficients(
              projector.projectForward(oldCoefficients.means),
              oldCoefficients.variancesOption.map(projector.projectForward))

            model.updateCoefficients(newCoefficients)
        }

        SingleNodeOptimizationProblem(
          configurationBroadcast.value,
          objectiveFunctionBuilder(projectedPriorModelOpt, projectedInterceptOpt),
          glmConstructorBroadcast.value,
          PhotonNonBroadcast(projectedNormalizationContext),
          varianceComputationType)
      }

    configurationBroadcast.unpersist()
    objectiveFunctionBuilderBroadcast.unpersist()
    glmConstructorBroadcast.unpersist()
    normalizationContextBroadcast.unpersist()

    new RandomEffectOptimizationProblem(optimizationProblems, glmConstructor)
  }
}
