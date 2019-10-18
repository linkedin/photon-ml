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
package com.linkedin.photon.ml.data

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import com.linkedin.photon.ml.Types.{FeatureShardId, UniqueSampleId}
import com.linkedin.photon.ml.constants.MathConst
import com.linkedin.photon.ml.data.scoring.CoordinateDataScores
import com.linkedin.photon.ml.spark.RDDLike

/**
 * Dataset implementation for fixed effect datasets.
 *
 * @param labeledPoints The input data
 * @param featureShardId The feature shard id
 */
protected[ml] class FixedEffectDataset(
    val labeledPoints: RDD[(UniqueSampleId, LabeledPoint)],
    val featureShardId: FeatureShardId)
  extends Dataset[FixedEffectDataset]
  with RDDLike {

  lazy val numFeatures: Int = labeledPoints.first()._2.features.length

  /**
   * Add scores to data offsets.
   *
   * @param scores The scores used throughout the coordinate descent algorithm
   * @return An updated dataset with scores added to offsets
   */
  override def addScoresToOffsets(scores: CoordinateDataScores): FixedEffectDataset = {

    // It's possible that other coordinates did not score some data. Since we're trying to add scores to the offset and
    // the default score is 0, the result of a left join vs. an inner join is the same. However, an inner join will drop
    // data which does not have a score. Thus, we need a left join.
    val updatedLabeledPoints = labeledPoints
      .leftOuterJoin(scores.scoresRdd)
      .mapValues { case (LabeledPoint(label, features, offset, weight), scoreOpt) =>
        LabeledPoint(label, features, offset + scoreOpt.getOrElse(MathConst.DEFAULT_SCORE), weight)
      }

    new FixedEffectDataset(updatedLabeledPoints, featureShardId)
  }

  /**
   * Get the Spark context.
   *
   * @return The Spark context
   */
  override def sparkContext: SparkContext = labeledPoints.sparkContext

  /**
   * Assign a given name to [[labeledPoints]].
   *
   * @note Not used to reference models in the logic of photon-ml, only used for logging currently.
   * @param name The parent name for all [[RDD]]s in this class
   * @return This object with the name of [[labeledPoints]] assigned
   */
  override def setName(name: String): FixedEffectDataset = {

    labeledPoints.setName(name)

    this
  }

  /**
   * Set the storage level of [[labeledPoints]], and persist their values across the cluster the first time they are
   * computed.
   *
   * @param storageLevel The storage level
   * @return This object with the storage level of [[labeledPoints]] set
   */
  override def persistRDD(storageLevel: StorageLevel): FixedEffectDataset = {

    if (!labeledPoints.getStorageLevel.isValid) labeledPoints.persist(storageLevel)

    this
  }

  /**
   * Mark [[labeledPoints]] as non-persistent, and remove all blocks for them from memory and disk.
   *
   * @return This object with [[labeledPoints]] marked non-persistent
   */
  override def unpersistRDD(): FixedEffectDataset = {

    if (labeledPoints.getStorageLevel.isValid) labeledPoints.unpersist()

    this
  }

  /**
   * Materialize [[labeledPoints]] (Spark [[RDD]]s are lazy evaluated: this method forces them to be evaluated).
   *
   * @return This object with [[labeledPoints]] materialized
   */
  override def materialize(): FixedEffectDataset = {

    labeledPoints.count()

    this
  }

  /**
   * Build a summary string for the dataset.
   *
   * @return A String representation of the dataset
   */
  override def toSummaryString: String = {

    val numSamples = labeledPoints.count()
    val weightSum = labeledPoints.values.map(_.weight).sum()
    val responseSum = labeledPoints.values.map(_.label).sum()
    val featureStats = labeledPoints.values.map(_.features.activeSize).stats()

    s"numSamples: $numSamples\n" +
      s"weightSum: $weightSum\n" +
      s"responseSum: $responseSum\n" +
      s"numFeatures: $numFeatures\n" +
      s"featureStats: $featureStats"
  }
}

object FixedEffectDataset {

  /**
   * Build an instance of a fixed effect dataset for the given feature shard.
   *
   * @param gameDataset The input dataset
   * @param featureShardId The feature shard ID
   * @return A new dataset with given configuration
   */
  protected[ml] def apply(
      gameDataset: RDD[(UniqueSampleId, GameDatum)],
      featureShardId: FeatureShardId): FixedEffectDataset = {

    val labeledPoints = gameDataset.mapValues(_.generateLabeledPointWithFeatureShardId(featureShardId))

    new FixedEffectDataset(labeledPoints, featureShardId)
  }
}
