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

import org.apache.spark.ml.linalg.{Vector => SparkVector}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.{DataTypes, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.storage.StorageLevel

import com.linkedin.photon.ml.Constants
import com.linkedin.photon.ml.Types.{FeatureShardId, UniqueSampleId}
import com.linkedin.photon.ml.data._
import com.linkedin.photon.ml.data.scoring.CoordinateDataScores
import com.linkedin.photon.ml.function.DistributedObjectiveFunction
import com.linkedin.photon.ml.model.{DatumScoringModel, FixedEffectModel}
import com.linkedin.photon.ml.optimization.{DistributedOptimizationProblem, FixedEffectOptimizationTracker, OptimizationTracker}
import com.linkedin.photon.ml.util.VectorUtils


/**
 * The optimization problem coordinate for a fixed effect model.
 *
 * @tparam Objective The type of objective function used to solve the fixed effect optimization problem
 * @param rawData The raw training data
 * @param optimizationProblem The fixed effect optimization problem
 */
protected[ml] class FixedEffectCoordinate[Objective <: DistributedObjectiveFunction](
    rawData: DataFrame,
    optimizationProblem: DistributedOptimizationProblem[Objective],
    featureShardId: FeatureShardId,
    inputColumnsNames: InputColumnsNames)
  extends Coordinate {

  var dataset: DataFrame =
    rawData.select(featureShardId, inputColumnsNames(InputColumnsNames.RESPONSE))


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
  override protected[algorithm] def trainModel(model: DatumScoringModel): (DatumScoringModel, OptimizationTracker) =
    model match {
      case fixedEffectModel: FixedEffectModel =>
        FixedEffectCoordinate.trainModel(
          dataset,
          optimizationProblem,
          featureShardId,
          Some(model))

      case _ =>
        throw new UnsupportedOperationException(
          s"Training model of type ${model.getClass} in ${this.getClass} is not supported")
    }


  /**
   * Compute scores for the coordinate dataset using the given model.
   *
   * @param model The input model
   * @return The dataset scores
   */
  override protected[algorithm] def score(model: DatumScoringModel): CoordinateDataScores = model match {

    case fixedEffectModel: FixedEffectModel =>
       FixedEffectCoordinate.score(dataset, fixedEffectModel)

    case _ =>
      throw new UnsupportedOperationException(
        s"Scoring with model of type ${model.getClass} in ${this.getClass} is not supported")
  }


  /**
   * Compute an optimized model (i.e. run the coordinate optimizer) for the current dataset.
   *
   * @return A (updated model, optimization state tracking information) tuple
   */
  override protected def trainModel(): (DatumScoringModel, OptimizationTracker) =
    FixedEffectCoordinate.trainModel(dataset, optimizationProblem, featureShardId, None)
}

object FixedEffectCoordinate {

  /**
   * Train a new [[FixedEffectModel]] (i.e. run model optimization).
   *
   * @param dataset The training dataset
   * @param optimizationProblem The optimization problem
   * @param featureShardId The ID of the feature shard for the training data
   * @param initialFixedEffectModelOpt An optional existing [[FixedEffectModel]] to use as a starting point for
   *                                   optimization
   * @return A new [[FixedEffectModel]]
   */
  private def trainModel[Function <: DistributedObjectiveFunction](
    dataset: DataFrame,
    optimizationProblem: DistributedOptimizationProblem[Function],
    featureShardId: FeatureShardId,
    initialFixedEffectModelOpt: Option[FixedEffectModel]): (FixedEffectModel, OptimizationTracker) = {

    val rdd = dataset
      .rdd
      .map { row =>
        val features = row.getAs[SparkVector](0)
        val label = row.getDouble(1)

        LabeledPoint(label, VectorUtils.mlToBreeze(features))
      }
    rdd.persist(StorageLevel.MEMORY_ONLY)

    val (glm, stateTracker) = initialFixedEffectModelOpt
      .map { initialFixedEffectModel =>
        optimizationProblem.runWithSampling(rdd, initialFixedEffectModel.model)
      }
      .getOrElse(optimizationProblem.runWithSampling(rdd))

    rdd.unpersist()

    (new FixedEffectModel(SparkSession.builder.getOrCreate.sparkContext.broadcast(glm), featureShardId),
      new FixedEffectOptimizationTracker(stateTracker))
  }


  /**
   * Score a dataset using a given [[FixedEffectModel]].
   *
   * @note The score is the dot product of the model coefficients with the feature values (i.e., it does not go
   *       through a non-linear link function).
   * @param fixedEffectDataset The dataset to score
   * @param fixedEffectModel The model used to score the dataset
   * @return The computed scores
   */
  protected[algorithm] def score(
    fixedEffectDataset: DataFrame,
    fixedEffectModel: FixedEffectModel): CoordinateDataScores = {

    //val modelBroadcast = fixedEffectModel.modelBroadcast
    //val scores = fixedEffectDataset.mapValues { features => modelBroadcast.value.computeScore(features)}
    //new CoordinateDataScores(scores)
    null
  }
}
