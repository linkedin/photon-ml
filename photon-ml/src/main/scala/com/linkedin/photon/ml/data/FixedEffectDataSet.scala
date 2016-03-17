/*
 * Copyright 2016 LinkedIn Corp. All rights reserved.
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

import com.linkedin.photon.ml.RDDLike


/**
 * DataSet implementation for fixed effect data sets
 *
 * @param labeledPoints the input data
 * @param featureShardId the feature shard id
 * @author xazhang
 */
protected[ml] class FixedEffectDataSet(val labeledPoints: RDD[(Long, LabeledPoint)], val featureShardId: String)
  extends DataSet[FixedEffectDataSet] with RDDLike {

  lazy val numFeatures = labeledPoints.first()._2.features.length

  /**
   * Add scores to data offsets
   *
   * @param scores the scores used throughout the coordinate descent algorithm
   */
  def addScoresToOffsets(scores: KeyValueScore): FixedEffectDataSet = {
    val updatedLabeledPoints = labeledPoints.leftOuterJoin(scores.scores)
        .mapValues { case (LabeledPoint(label, features, offset, weight), scoreOption) =>
      LabeledPoint(label, features, offset + scoreOption.getOrElse(0.0), weight)
    }
    new FixedEffectDataSet(updatedLabeledPoints, featureShardId)
  }

  override def sparkContext: SparkContext = labeledPoints.sparkContext

  override def setName(name: String): this.type = {
    labeledPoints.setName(name)
    this
  }

  override def persistRDD(storageLevel: StorageLevel): this.type = {
    if (!labeledPoints.getStorageLevel.isValid) labeledPoints.persist(storageLevel)
    this
  }

  override def unpersistRDD(): this.type = {
    if (labeledPoints.getStorageLevel.isValid) labeledPoints.unpersist()
    this
  }

  override def materialize(): this.type = {
    labeledPoints.count()
    this
  }

  /**
   * Build a summary string for the dataset
   *
   * @return string representation
   */
  override def toSummaryString: String = {
    val numSamples = labeledPoints.count()
    val weightSum = labeledPoints.values.map(_.weight).sum()
    val responseSum = labeledPoints.values.map(_.label).sum()
    val featureStats = labeledPoints.values.map(_.features.activeSize).stats()
    s"numSamples: $numSamples\nweightSum: $weightSum\nresponseSum: $responseSum" +
        s"\nnumFeatures: $numFeatures\nfeatureStats: $featureStats"
  }
}

object FixedEffectDataSet {

  /**
   * Build an instance of fixed effect dataset with the given configuration
   *
   * @param gameDataSet the input dataset
   * @param fixedEffectDataConfiguration the data configuration
   * @return new dataset with given configuration
   */
  protected[ml] def buildWithConfiguration(
      gameDataSet: RDD[(Long, GameData)],
      fixedEffectDataConfiguration: FixedEffectDataConfiguration): FixedEffectDataSet = {

    val featureShardId = fixedEffectDataConfiguration.featureShardId
    val labeledPoints = gameDataSet.mapValues(_.generateLabeledPointWithFeatureShardId(featureShardId))
    new FixedEffectDataSet(labeledPoints, featureShardId)
  }
}
