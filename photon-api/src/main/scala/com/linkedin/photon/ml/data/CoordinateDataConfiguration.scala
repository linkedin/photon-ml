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

import com.linkedin.photon.ml.Types._

/**
 * Generic trait for a configuration to define a dataset.
 */
sealed trait CoordinateDataConfiguration {

  def featureShardId: FeatureShardId
  def minNumPartitions: Int

  require(0 < minNumPartitions, s"Cannot have fewer than 1 Spark partition: $minNumPartitions")
}

/**
 * Configuration needed in order to generate a [[com.linkedin.photon.ml.data.FixedEffectDataset]].
 *
 * @param featureShardId Key of the feature shard used to generate the dataset
 * @param minNumPartitions Minimum number of data partitions
 */
case class FixedEffectDataConfiguration(
    featureShardId: FeatureShardId,
    minNumPartitions: Int = 1)
  extends CoordinateDataConfiguration

/**
 * Configurations needed in order to generate a [[com.linkedin.photon.ml.data.RandomEffectDataset]].
 *
 * @param randomEffectType The corresponding random effect type of the dataset
 * @param featureShardId Key of the feature shard used to generate the dataset
 * @param minNumPartitions Minimum number of data partitions
 * @param numActiveDataPointsLowerBound The lower bound on the number of samples required to train a random effect model
 *                                      for an entity. If this bound is not met, the data is discarded.
 * @param numActiveDataPointsUpperBound The upper bound on the number of samples to keep (via reservoir sampling) as
 *                                      "active" for each per-entity local dataset. The remaining samples will
 *                                      be kept as "passive" data. "Active" data is used for model training and residual
 *                                      computation. "Passive" data is used only for residual computation.
 * @param numFeaturesToSamplesRatioUpperBound The upper bound on the ratio between number of features and number of
 *                                            samples. Used for dimensionality reduction for IDs with very few samples.
 */
case class RandomEffectDataConfiguration(
    randomEffectType: REType,
    featureShardId: FeatureShardId,
    minNumPartitions: Int = 1,
    numActiveDataPointsLowerBound: Option[Int] = None,
    numActiveDataPointsUpperBound: Option[Int] = None,
    numFeaturesToSamplesRatioUpperBound: Option[Double] = None)
  extends CoordinateDataConfiguration {

  require(
    numActiveDataPointsLowerBound.forall(_ > 0),
    s"Active data lower bound must be greater than 0: ${numActiveDataPointsLowerBound.get}")
  require(
    numActiveDataPointsUpperBound.forall(_ > 0),
    s"Active data upper bound must be greater than 0: $numActiveDataPointsUpperBound")
  require(
    numActiveDataPointsLowerBound.forall(_ <= numActiveDataPointsUpperBound.getOrElse(Int.MaxValue)),
    s"Active data lower bound must be less than active data upper bound (${numActiveDataPointsUpperBound.get})")
  require(
    numFeaturesToSamplesRatioUpperBound.forall(_ > 0),
    s"Features to samples ratio must be greater than 0: ${numFeaturesToSamplesRatioUpperBound.get}")
}
