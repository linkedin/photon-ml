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
package com.linkedin.photon.ml.model

import org.apache.spark.rdd.RDD
import org.apache.spark.rdd.RDD._
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{HashPartitioner, SparkContext}

import com.linkedin.photon.ml.TaskType.TaskType
import com.linkedin.photon.ml.Types.{UniqueSampleId, REId, REType, FeatureShardId}
import com.linkedin.photon.ml.data.GameDatum
import com.linkedin.photon.ml.data.scoring.{CoordinateDataScores, ModelDataScores}
import com.linkedin.photon.ml.spark.RDDLike
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel

/**
 * Representation of a random effect model.
 *
 * @param modelsRDD The models, one for each unique random effect value
 * @param randomEffectType The random effect type
 * @param featureShardId The feature shard id
 */
class RandomEffectModel(
    protected[ml] val modelsRDD: RDD[(REId, GeneralizedLinearModel)],
    val randomEffectType: REType,
    val featureShardId: FeatureShardId)
  extends DatumScoringModel
  with RDDLike {

  override val modelType: TaskType = RandomEffectModel.determineModelType(modelsRDD)

  //
  // RandomEffectModel functions
  //

  /**
   * Create a new [[RandomEffectModel]] with new underlying models.
   *
   * @param newModelsRdd The new underlying models, one per entity
   * @return A new [[RandomEffectModel]]
   */
  def update(newModelsRdd: RDD[(REId, GeneralizedLinearModel)]): RandomEffectModel =
    new RandomEffectModel(newModelsRdd, randomEffectType, featureShardId)

  //
  // DatumScoringModel functions
  //

  /**
   * Compute the score for the dataset.
   *
   * @note Use a static method to avoid serializing entire model object during RDD operations.
   * @param dataPoints The dataset to score (Note that the Long in the RDD is a unique identifier for the paired
   *                   [[GameDatum]] object, referred to in the GAME code as the "unique id")
   * @return The computed scores
   */
  override def score(dataPoints: RDD[(UniqueSampleId, GameDatum)]): ModelDataScores =
    RandomEffectModel.score(
      dataPoints,
      modelsRDD,
      randomEffectType,
      featureShardId,
      ModelDataScores.toScore,
      ModelDataScores.apply)

  /**
   * Compute the scores for the GAME dataset, and store the scores only.
   *
   * @note Use a static method to avoid serializing entire model object during RDD operations.
   * @param dataPoints The dataset to score (Note that the Long in the RDD is a unique identifier for the paired
   *                   [[GameDatum]] object, referred to in the GAME code as the "unique id")
   * @return The computed scores
   */
  override def scoreForCoordinateDescent(dataPoints: RDD[(UniqueSampleId, GameDatum)]): CoordinateDataScores =
    RandomEffectModel.score(
      dataPoints,
      modelsRDD,
      randomEffectType,
      featureShardId,
      CoordinateDataScores.toScore,
      CoordinateDataScores.apply)

  //
  // Summarizable functions
  //

  /**
   * Summarize this model in text format.
   *
   * @return A model summary in String representation
   */
  override def toSummaryString: String = {

    val stringBuilder = new StringBuilder("Random Effect Model:")

    stringBuilder.append(s"\nRandom Effect Type: '$randomEffectType'")
    stringBuilder.append(s"\nFeature Shard ID: '$featureShardId'")
    stringBuilder.append(s"\nLength: ${modelsRDD.values.map(_.coefficients.means.length).stats()}")
    stringBuilder.append(s"\nMean: ${modelsRDD.values.map(_.coefficients.meansL2Norm).stats()}")
    if (modelsRDD.first()._2.coefficients.variancesOption.isDefined) {
      stringBuilder.append(s"\nVariance: ${modelsRDD.values.map(_.coefficients.variancesL2NormOption.get).stats()}")
    }

    stringBuilder.toString()
  }

  //
  // RDDLike functions
  //

  /**
   * Get the Spark context.
   *
   * @return The Spark context
   */
  override protected[ml] def sparkContext: SparkContext = modelsRDD.sparkContext

  /**
   * Assign a given name to [[modelsRDD]].
   *
   * @note Not used to reference models in the logic of photon-ml, only used for logging currently.
   * @param name The parent name for all [[RDD]]s in this class
   * @return This object with the name of [[modelsRDD]] assigned
   */
  override protected[ml] def setName(name: String): RandomEffectModel = {

    modelsRDD.setName(name)

    this
  }

  /**
   * Set the storage level of [[modelsRDD]], and persist their values across the cluster the first time they are
   * computed.
   *
   * @param storageLevel The storage level
   * @return This object with the storage level of [[modelsRDD]] set
   */
  override protected[ml] def persistRDD(storageLevel: StorageLevel): RandomEffectModel = {

    if (!modelsRDD.getStorageLevel.isValid) modelsRDD.persist(storageLevel)

    this
  }

  /**
   * Mark [[modelsRDD]] as non-persistent, and remove all blocks for them from memory and disk.
   *
   * @return This object with [[modelsRDD]] marked non-persistent
   */
  override protected[ml] def unpersistRDD(): RandomEffectModel = {

    if (modelsRDD.getStorageLevel.isValid) modelsRDD.unpersist()

    this
  }

  /**
   * Materialize [[modelsRDD]] (Spark [[RDD]]s are lazy evaluated: this method forces them to be evaluated).
   *
   * @return This object with [[modelsRDD]] materialized
   */
  override protected[ml] def materialize(): RandomEffectModel = {

    modelsRDD.count()

    this
  }

  /**
   * Compares two [[RandomEffectModel]] objects.
   *
   * @param that Some other object
   * @return True if the models have the same types and the same underlying models for each random effect ID, false
   *         otherwise
   */
  override def equals(that: Any): Boolean = that match {

    case other: RandomEffectModel =>
      val sameMetaData = this.randomEffectType == other.randomEffectType && this.featureShardId == other.featureShardId

      lazy val sameCoefficientsRDD = this
        .modelsRDD
        .fullOuterJoin(other.modelsRDD)
        .mapPartitions { iterator =>
          Iterator.single(iterator.forall { case (_, (modelOpt1, modelOpt2)) =>
            modelOpt1.isDefined && modelOpt2.isDefined && modelOpt1.get.equals(modelOpt2.get)
          })
        }
        .filter(!_)
        .count() == 0

      sameMetaData && sameCoefficientsRDD

    case _ => false
  }

  /**
   * Returns a hash code value for the object.
   *
   * TODO: Violation of the hashCode() contract
   *
   * @return An [[Int]] hash code
   */
  override def hashCode(): Int = super.hashCode()
}

object RandomEffectModel {

  /**
   * Determine the random effect model type: even though the model has many sub-problems, there is only one loss
   * function type for a given random effect model.
   *
   * TODO: We should consider refactoring this method to instead take a TaskType and verify that all sub-models match
   *       that type - it will be faster for large numbers of random effect models. Note that it may still be a
   *       bottleneck if we check each time a new RandomEffectModel is created.
   *
   * @param modelsRDD The random effect models
   * @return The GAME model type
   */
  protected def determineModelType(modelsRDD: RDD[(REId, GeneralizedLinearModel)]): TaskType = {

    val modelTypes = modelsRDD.values.map(_.modelType).distinct().collect()

    require(
      modelTypes.length == 1,
      s"${modelsRDD.name} has multiple model types:\n${modelTypes.mkString(", ")}")

    modelTypes.head
  }

  /**
   * Compute the scores for a dataset, using random effect models.
   *
   * @param dataPoints The dataset to score
   * @param modelsRDD The individual random effect models to use for scoring
   * @param randomEffectType The random effect type
   * @param featureShardId The feature shard id
   * @return The scores
   */
  private def score[T, V](
      dataPoints: RDD[(UniqueSampleId, GameDatum)],
      modelsRDD: RDD[(REId, GeneralizedLinearModel)],
      randomEffectType: REType,
      featureShardId: FeatureShardId,
      toScore: (GameDatum, Double) => T,
      toResult: RDD[(UniqueSampleId, T)] => V): V = {

    val hashPartitioner = new HashPartitioner(dataPoints.getNumPartitions)

    /*
     * We perform a replicated partitioned hash join here under the assumption that we can fit the per partition
     * random effect models in memory. We first partition both relations using the same partitioner and then zip them.
     * This ensures that the same keys from both relations go in the same partition. Given above, we can now perform the
     * join by doing the following operations per partition:
     *   1. Load the random effect models in memory
     *   2. Iterate over the data points
     *   3. For each data point, look up the corresponding random effect model in the in memory map and score
     */
    val scores = dataPoints
      .map { case (uniqueId, gameDatum) =>
        (gameDatum.idTagToValueMap(randomEffectType), (uniqueId, gameDatum))
      }
      .partitionBy(hashPartitioner)
      .zipPartitions(modelsRDD.partitionBy(hashPartitioner)) { (dataIt, modelIt) =>

        val lookupTable = modelIt.toMap

        dataIt.map { case (id, (uid, datum)) =>
          val score = lookupTable
            .get(id)
            .map(_.computeScore(datum.featureShardContainer(featureShardId)))
            .getOrElse(0.0)

          (uid, toScore(datum, score))
        }
      }

    toResult(scores)
  }
}
