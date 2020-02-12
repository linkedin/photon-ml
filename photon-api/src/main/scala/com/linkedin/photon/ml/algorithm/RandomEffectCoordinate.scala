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

import org.apache.spark.SparkContext
import org.apache.spark.ml.linalg.{Vector => SparkVector}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{DataTypes, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SparkSession, functions}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.sql.functions.col
import com.linkedin.photon.ml.normalization.NormalizationContext
import com.linkedin.photon.ml.Constants
import com.linkedin.photon.ml.Types.{FeatureShardId, REType}
import com.linkedin.photon.ml.data._
import com.linkedin.photon.ml.data.scoring.CoordinateDataScores
import com.linkedin.photon.ml.function.SingleNodeObjectiveFunction
import com.linkedin.photon.ml.model.{Coefficients, DatumScoringModel, RandomEffectModel}
import com.linkedin.photon.ml.optimization.VarianceComputationType.VarianceComputationType
import com.linkedin.photon.ml.optimization.game.{RandomEffectOptimizationConfiguration, RandomEffectOptimizationProblem}
import com.linkedin.photon.ml.optimization._
import com.linkedin.photon.ml.spark.RDDLike
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import com.linkedin.photon.ml.util.VectorUtils
/**
 * The optimization problem coordinate for a random effect model.
 *
 * @param rEType
 * @param rawData    The raw training dataframe
 * @tparam Objective The type of objective function used to solve individual random effect optimization problems
 * @param optimizationProblem The random effect optimization problem
 * @param inputColumnsNames
 */
protected[ml] class RandomEffectCoordinate[Objective <: SingleNodeObjectiveFunction](
  rEType: REType,
  rawData: DataFrame,
  optimizationProblem: RandomEffectOptimizationProblem[Objective],
  featureShardId: FeatureShardId,
  inputColumnsNames: InputColumnsNames)
  extends Coordinate
    with RDDLike {

  /* Get the training data from raw data */
  var dataset: DataFrame = {
    val label =  inputColumnsNames(InputColumnsNames.RESPONSE)
    val offset = inputColumnsNames(InputColumnsNames.OFFSET)
    val weight = inputColumnsNames(InputColumnsNames.WEIGHT)

    rawData
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
  override protected def updateDataset(scores: CoordinateDataScores) = {

    // TODO: change scores to dataframe
    val schemaFields = Array[StructField](
      StructField(Constants.UNIQUE_SAMPLE_ID, DataTypes.LongType, nullable = false),
      StructField("score", DataTypes.DoubleType, nullable = false))
    dataset = SparkSession
      .builder
      .getOrCreate
      .createDataFrame(scores.scoresRdd.map(Row.fromTuple(_)), new StructType(schemaFields))
      .join(rawData, Constants.UNIQUE_SAMPLE_ID)
      // TODO: WHAT IF OFFSET DOESN'T EXIST
      //.withColumnRenamed("score", inputColumnsNames(InputColumnsNames.OFFSET))
      .withColumn(inputColumnsNames(InputColumnsNames.OFFSET),
                  col(inputColumnsNames(InputColumnsNames.OFFSET)) + col("score"))
  }

  /**
   * Compute an optimized model (i.e. run the coordinate optimizer) for the current dataset using an existing model as
   * a starting point.
   *
   * @param model The model to use as a starting point
   * @return A (updated model, optional optimization tracking information) tuple
   */
  override protected[algorithm] def trainModel(
      model: DatumScoringModel): (DatumScoringModel, OptimizationTracker) =

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

  /**
   * Compute an optimized model (i.e. run the coordinate optimizer) for the current dataset.
   *
   * @return A (updated model, optimization state tracking information) tuple
   */
  override protected[algorithm] def trainModel(): (DatumScoringModel, OptimizationTracker) = {

    val (newModel, optimizationTracker) = RandomEffectCoordinate.trainModel(
      dataset,
      rEType,
      featureShardId,
      optimizationProblem,
      None)

    (newModel, optimizationTracker)
  }

  /**
   * Compute scores for the coordinate data using a given model.
   *
   * @param model The input model
   * @return The dataset scores
   */
  override protected[algorithm] def score(model: DatumScoringModel): CoordinateDataScores = model match {
    case randomEffectModel: RandomEffectModel =>
      RandomEffectCoordinate.score(dataset, randomEffectModel)

    case _ =>
      throw new UnsupportedOperationException(
        s"Scoring with model of type ${model.getClass} in ${this.getClass} is not supported")
  }

  //
  // RDDLike Functions
  //

  /**
   * Get the Spark context.
   *
   * @return The Spark context
   */
  override def sparkContext: SparkContext = optimizationProblem.sparkContext

  /**
   * Assign a given name to the [[optimizationProblem]] [[RDD]].
   *
   * @param name The parent name for all [[RDD]] objects in this class
   * @return This object with the name of the [[optimizationProblem]] [[RDD]] assigned
   */
  override def setName(name: String): RandomEffectCoordinate[Objective] = {

    optimizationProblem.setName(name)

    this
  }

  /**
   * Set the persistence storage level of the [[optimizationProblem]] [[RDD]].
   *
   * @param storageLevel The storage level
   * @return This object with the storage level of the [[optimizationProblem]] [[RDD]] set
   */
  override def persistRDD(storageLevel: StorageLevel): RandomEffectCoordinate[Objective] = {

    optimizationProblem.persistRDD(storageLevel)

    this
  }

  /**
   * Mark the [[optimizationProblem]] [[RDD]] as unused, and asynchronously remove all blocks for it from memory and
   * disk.
   *
   * @return This object with the [[optimizationProblem]] [[RDD]] unpersisted
   */
  override def unpersistRDD(): RandomEffectCoordinate[Objective] = {

    optimizationProblem.unpersistRDD()

    this
  }

  /**
   * Materialize the [[optimizationProblem]] [[RDD]] (Spark [[RDD]]s are lazy evaluated: this method forces them to be
   * evaluated).
   *
   * @return This object with the [[optimizationProblem]] [[RDD]] materialized
   */
  override def materialize(): RandomEffectCoordinate[Objective] = {

    optimizationProblem.materialize()

    this
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
    val randomEffectOptimizationProblem = RandomEffectOptimizationProblem(
      data,
      rEType,
      configuration,
      objectiveFunctionFactory,
      glmConstructor,
      normalizationContext,
      varianceComputationType,
      interceptIndexOpt)

    new RandomEffectCoordinate(rEType, data, randomEffectOptimizationProblem, featureShardId, inputColumnsNames)
  }

  /**
   * Train a new [[RandomEffectModel]] (i.e. run model optimization for each entity).
   *
   * @tparam Function The type of objective function used to solve individual random effect optimization problems
   * @param randomEffectDataset The training dataset
   * @param randomEffectType
   * @param featureShardId
   * @param randomEffectOptimizationProblem The per-entity optimization problems
   * @param initialRandomEffectModelOpt An optional existing [[RandomEffectModel]] to use as a starting point for
   *                                    optimization
   * @return A (new [[RandomEffectModel]], optional optimization stats) tuple
   */
  protected[algorithm] def trainModel[Function <: SingleNodeObjectiveFunction](
    randomEffectDataset: DataFrame,
    randomEffectType: REType,
    featureShardId: FeatureShardId,
    randomEffectOptimizationProblem: RandomEffectOptimizationProblem[Function],
    initialRandomEffectModelOpt: Option[RandomEffectModel]): (RandomEffectModel, RandomEffectOptimizationTracker) = {

    val rdd = randomEffectDataset
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

        (reid, LocalDataset(result.toArray))
      }

    // TODO: remove pre-REID optimization problems
    // All 3 RDDs involved in these joins use the same partitioner
    val dataAndOptimizationProblems = rdd.join(randomEffectOptimizationProblem.optimizationProblems)

    // Left join the models to data and optimization problems for cases where we have a prior model but no new data
    val (newModels, randomEffectOptimizationTracker) = initialRandomEffectModelOpt
      .map { randomEffectModel =>
        val modelsAndTrackers = randomEffectModel
          .modelsRDD
          .leftOuterJoin(dataAndOptimizationProblems)
          .mapValues {
            case (localModel, Some((localDataset, optimizationProblem))) =>
              val trainingLabeledPoints = localDataset.dataPoints
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
        val modelsAndTrackers = dataAndOptimizationProblems
          .mapValues { case (localDataset, optimizationProblem) =>
            val trainingLabeledPoints = localDataset.dataPoints
            optimizationProblem.run(trainingLabeledPoints)
          }
        modelsAndTrackers.persist(StorageLevel.MEMORY_AND_DISK_SER)

        val models = modelsAndTrackers.mapValues(_._1)
        val optimizationTracker = RandomEffectOptimizationTracker(modelsAndTrackers.map(_._2._2))
        (models, optimizationTracker)
      }

    val newRandomEffectModel = new RandomEffectModel(
      newModels,
      randomEffectType,
      featureShardId)

    (newRandomEffectModel, randomEffectOptimizationTracker)
  }


  /**
   * Score a dataset using a given [[RandomEffectModel]].
   *
   * For information about the differences between active and passive data
   * documentation.
   *
   * @note The score is the raw dot product of the model coefficients and the feature values - it does not go through a
   *       non-linear link function.
   * @param randomEffectDataset The data set to score
   * @param randomEffectModel The [[RandomEffectModel]] with which to score
   * @return The computed scores
   */
  protected[algorithm] def score(
      randomEffectDataset: DataFrame,
      randomEffectModel: RandomEffectModel): CoordinateDataScores = {

    /*
    // There may be more models than active data. However, since we're computing residuals for future coordinates, no
    // data means no residual. Therefore, we use an inner join. Note that the active data and models use the same
    // partitioner, but scores need to use GameDatum partitioner.
    val activeScores = randomEffectDataset
      .activeData
      .join(randomEffectModel.modelsRDD)
      .flatMap { case (_, (localDataset, model)) =>
        localDataset.dataPoints.map { case (uniqueId, labeledPoint) =>
          (uniqueId, model.computeScore(labeledPoint.features))
        }
      }
      .partitionBy(randomEffectDataset.uniqueIdPartitioner)

    // Passive data already uses the GameDatum partitioner. Note that this code assumes few (if any) entities have a
    // passive dataset.
    val passiveDataREIds = randomEffectDataset.passiveDataREIds
    val modelsForPassiveData = randomEffectModel
      .modelsRDD
      .filter { case (reId, _) =>
        passiveDataREIds.value.contains(reId)
      }
      .collectAsMap()
    val passiveScores = randomEffectDataset
      .passiveData
      .mapValues { case (randomEffectId, labeledPoint) =>
        modelsForPassiveData(randomEffectId).computeScore(labeledPoint.features)
      }

    new CoordinateDataScores(activeScores ++ passiveScores)

     */
    return null
  }
}
