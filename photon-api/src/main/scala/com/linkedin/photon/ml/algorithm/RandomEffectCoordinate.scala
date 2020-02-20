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

import scala.collection.mutable

import org.apache.spark.ml.linalg.{Vector => SparkVector}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.{DataFrame, functions}
import org.apache.spark.storage.StorageLevel

import com.linkedin.photon.ml.Types.{FeatureShardId, REType}
import com.linkedin.photon.ml.data._
import com.linkedin.photon.ml.function.SingleNodeObjectiveFunction
import com.linkedin.photon.ml.model.{Coefficients, DatumScoringModel, RandomEffectModel}
import com.linkedin.photon.ml.normalization.NormalizationContext
import com.linkedin.photon.ml.optimization.VarianceComputationType.VarianceComputationType
import com.linkedin.photon.ml.optimization._
import com.linkedin.photon.ml.optimization.game.RandomEffectOptimizationConfiguration
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import com.linkedin.photon.ml.util.{ApiUtils, PhotonNonBroadcast, VectorUtils}

/**
 * The optimization problem coordinate for a random effect model.
 *
 * @tparam Objective The type of objective function used to solve individual random effect optimization problems
 * @param rEType  The random effect type
 * @param rawData    The raw training dataframe
 * @param optimizationProblem The single node optimization problem
 * @param inputColumnsNames
 */
protected[ml] class RandomEffectCoordinate[Objective <: SingleNodeObjectiveFunction](
  rEType: REType,
  var rawData: DataFrame,
  optimizationProblem: SingleNodeOptimizationProblem[Objective],
  featureShardId: FeatureShardId,
  inputColumnsNames: InputColumnsNames)
  extends Coordinate {

  /* Get the training data from raw data */
  var dataset: DataFrame = null

  protected def updateDataset(): Unit = {

    val label = inputColumnsNames(InputColumnsNames.RESPONSE)
    val offset = inputColumnsNames(InputColumnsNames.OFFSET)
    val weight = inputColumnsNames(InputColumnsNames.WEIGHT)

    dataset = rawData
      .select(rEType, featureShardId, label, offset, weight)
      .groupBy(rEType)
      .agg(
        functions.collect_list(featureShardId),
        functions.collect_list(label),
        functions.collect_list(offset),
        functions.collect_list(weight))
  }

  //
  // Coordinate functions
  //
  override protected def updateOffset(model: DatumScoringModel) = {

    model match {
      case randomEffectModel: RandomEffectModel =>
        rawData = RandomEffectCoordinate.updateOffset(
          rawData, randomEffectModel, featureShardId,
          rEType, inputColumnsNames)

        updateDataset()

      case _ =>
        throw new UnsupportedOperationException(s"Unsupported model type: ${model.modelType}")
    }
  }

  /**
   * Compute an optimized model (i.e. run the coordinate optimizer) for the current dataset using an existing model as
   * a starting point.
   *
   * @param model The model to use as a starting point
   * @return A (updated model, optional optimization tracking information) tuple
   */
  override protected[algorithm] def trainModel(
      model: DatumScoringModel): (DatumScoringModel, OptimizationTracker) = {

    if (dataset == null) {
      updateDataset()
    }

    model match {
      case randomEffectModel: RandomEffectModel =>
        val (newModel, optimizationTracker) = RandomEffectCoordinate.trainModel(
          dataset,
          rEType,
          featureShardId,
          optimizationProblem,
          Some(randomEffectModel))

        (newModel, optimizationTracker)
      case _ =>
        throw new UnsupportedOperationException(
          s"Updating model of type ${model.getClass} in ${this.getClass} is not supported")
    }
  }

  /**
   * Compute an optimized model (i.e. run the coordinate optimizer) for the current dataset.
   *
   * @return A (updated model, optimization state tracking information) tuple
   */
  override protected[algorithm] def trainModel(): (DatumScoringModel, OptimizationTracker) = {
    if (dataset == null) {
      updateDataset()
    }

    val (newModel, optimizationTracker) = RandomEffectCoordinate.trainModel(
      dataset,
      rEType,
      featureShardId,
      optimizationProblem,
      None)

    (newModel, optimizationTracker)
  }

}

object RandomEffectCoordinate {

  /**
   * Helper function to construct [[RandomEffectCoordinate]] objects.
   *
   * @tparam RandomEffectObjective The type of objective function used to solve individual random effect optimization problems
   * @param data  The data on which to run the optimization algorithm
   * @param rEType
   * @param featureShardId
   * @param inputColumnsNames
   * @param configuration The optimization problem configuration
   * @param objectiveFunctionFactory The objective function to optimize
   * @param glmConstructor The function to use for producing GLMs from trained coefficients
   * @param normalizationContext The normalization context
   * @param varianceComputationType If and how coefficient variances should be computed
   * @param interceptIndexOpt The index of the intercept, if there is one
   * @return A new [[RandomEffectCoordinate]] object
   */
  protected[ml] def apply[RandomEffectObjective <: SingleNodeObjectiveFunction](
    data: DataFrame,
    rEType: REType,
    featureShardId: FeatureShardId,
    inputColumnsNames: InputColumnsNames,
    configuration: RandomEffectOptimizationConfiguration,
    objectiveFunctionFactory: Option[Int] => RandomEffectObjective,
    glmConstructor: Coefficients => GeneralizedLinearModel,
    normalizationContext: NormalizationContext,
    varianceComputationType: VarianceComputationType,
    interceptIndexOpt: Option[Int] = None): RandomEffectCoordinate[RandomEffectObjective] = {

    // Generate parameters of ProjectedRandomEffectCoordinate
    val optimizationProblem = SingleNodeOptimizationProblem(
      configuration,
      objectiveFunctionFactory(interceptIndexOpt),
      glmConstructor,
      PhotonNonBroadcast(normalizationContext),
      varianceComputationType)

    new RandomEffectCoordinate(rEType, data, optimizationProblem, featureShardId, inputColumnsNames)
  }

  /**
   * Train a new [[RandomEffectModel]] (i.e. run model optimization for each entity).
   *
   * @tparam Function The type of objective function used to solve individual random effect optimization problems
   * @param randomEffectDataset The training dataset
   * @param randomEffectType
   * @param featureShardId
   * @param optimizationProblem The per-entity optimization problems
   * @param initialRandomEffectModelOpt An optional existing [[RandomEffectModel]] to use as a starting point for
   *                                    optimization
   * @return A (new [[RandomEffectModel]], optional optimization stats) tuple
   */
  protected[algorithm] def trainModel[Function <: SingleNodeObjectiveFunction](
    randomEffectDataset: DataFrame,
    randomEffectType: REType,
    featureShardId: FeatureShardId,
    optimizationProblem: SingleNodeOptimizationProblem[Function],
    initialRandomEffectModelOpt: Option[RandomEffectModel]): (RandomEffectModel, RandomEffectOptimizationTracker) = {

    val data = randomEffectDataset
      .rdd
      .map { row =>
        val reid = row.getInt(0).toString
        val features = row.getList[SparkVector](1)
        val labels = row.getList[Double](2)
        val offsets = row.getList[Double](3)
        val weights = row.getList[Double](4)

        val fIter = features.iterator()
        val lIter = labels.iterator()
        val oIter = offsets.iterator()
        val wIter = weights.iterator()

        require(features.size == labels.size)
        require(features.size == offsets.size)
        require(features.size == weights.size)

        val result = new mutable.ArrayBuffer[LabeledPoint](features.size)

        (0 until features.size).map { _ =>
          result += LabeledPoint(lIter.next(), VectorUtils.mlToBreeze(fIter.next()), oIter.next(), wIter.next())
        }

        (reid, result.toArray)
      }

    // Left join the models to data and optimization problems for cases where we have a prior model but no new data
    val (newModels, randomEffectOptimizationTracker) = initialRandomEffectModelOpt
      .map { randomEffectModel =>
        val modelsRdd = randomEffectModel.toRDD()
        val modelsAndTrackers = modelsRdd
          .leftOuterJoin(data)
          .mapValues {
            case (localModel, Some((localDataset))) =>
              val trainingLabeledPoints = localDataset
              val (updatedModel, stateTrackers) = optimizationProblem.run(trainingLabeledPoints, localModel)

              (updatedModel, Some(stateTrackers))

            case (localModel, _) =>
              (localModel, None)
          }
        modelsAndTrackers.persist(StorageLevel.MEMORY_ONLY_SER)

        val models = modelsAndTrackers.mapValues(_._1)
        val optimizationTracker = RandomEffectOptimizationTracker(modelsAndTrackers.flatMap(_._2._2))

        (models, optimizationTracker)
      }
      .getOrElse {
        val modelsAndTrackers = data.mapValues (optimizationProblem.run(_))
        modelsAndTrackers.persist(StorageLevel.MEMORY_AND_DISK_SER)

        val models = modelsAndTrackers.mapValues(_._1)
        val optimizationTracker = RandomEffectOptimizationTracker(modelsAndTrackers.map(_._2._2))
        (models, optimizationTracker)
      }

    val newRandomEffectModel = new RandomEffectModel(
      RandomEffectModel.toDataFrame(newModels),
      randomEffectType,
      featureShardId)

    (newRandomEffectModel, randomEffectOptimizationTracker)
  }

  def getScoreFieldName(rEType: REType): String = {
    return s"${rEType}_score"
  }

  /**
   * Score a dataset using a given [[RandomEffectModel]].
   *
   * For information about the differences between active and passive data
   * documentation.
   *
   * @note The score is the raw dot product of the model coefficients and the feature values - it does not go through a
   *       non-linear link function.
   * @param dataset The data set to score
   * @param randomEffectModel The [[RandomEffectModel]] with which to score
   * @return The computed scores
   */
  def updateOffset(
    dataset: DataFrame, randomEffectModel: RandomEffectModel, featureShardId: FeatureShardId,
    rEType: REType,
    inputColumnsNames: InputColumnsNames): DataFrame = {

    require(
      featureShardId == randomEffectModel.featureShardId,
      s"Random effect coordinate featureShardId ${featureShardId} != model.featureShardId ${
        randomEffectModel
          .featureShardId
      }")

    require(
      rEType == randomEffectModel.randomEffectType,
      s"Random effect coordinate randomEffectType ${rEType} != model.randomEffectType ${
        randomEffectModel
          .randomEffectType
      }")

    val scoreField = getScoreFieldName(rEType)
    val offset = inputColumnsNames(InputColumnsNames.OFFSET)
    val hasOffsetField = ApiUtils.hasColumn(dataset, offset)
    val hasCoordinateScoreField = ApiUtils.hasColumn(dataset, scoreField)

    if (hasOffsetField && hasCoordinateScoreField) {
      // offset = offset - old_coordinateScore + new_coordinateScore
      dataset.withColumn(offset, col(offset) - col(scoreField))
      randomEffectModel.computeScore(dataset, scoreField)
        .withColumn(offset, col(offset) + col(scoreField))
    } else if (!hasOffsetField && !hasCoordinateScoreField) {
      randomEffectModel.computeScore(dataset, scoreField)
        .withColumn(offset, col(scoreField))
    } else if (hasOffsetField && !hasCoordinateScoreField) {
      randomEffectModel.computeScore(dataset, scoreField)
        .withColumn(offset, col(offset) + col(scoreField))
    } else {
      throw new UnsupportedOperationException("It shouldn't happen!")
    }
  }
}
