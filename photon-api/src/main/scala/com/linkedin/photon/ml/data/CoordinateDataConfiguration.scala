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
import com.linkedin.photon.ml.projector.{IndexMapProjection, ProjectorType}

/**
 * Generic trait for a configuration to define a coordinate dataset.
 */
sealed trait CoordinateDataConfiguration {

  def featureShardId: FeatureShardId
  def minNumPartitions: Int

  require(0 < minNumPartitions, s"Cannot have fewer than 1 Spark partition: $minNumPartitions")
}

/**
 * Configuration needed in order to generate a [[com.linkedin.photon.ml.data.FixedEffectDataSet]].
 *
 * @param featureShardId Key of the feature shard used to generate the dataset
 * @param minNumPartitions Minimum number of data partitions
 */
case class FixedEffectDataConfiguration(
    featureShardId: FeatureShardId,
    minNumPartitions: Int = 1)
  extends CoordinateDataConfiguration

/**
 * Configurations needed in order to generate a [[com.linkedin.photon.ml.data.RandomEffectDataSet]]
 *
 * @param randomEffectType The corresponding random effect type of the dataset
 * @param featureShardId Key of the feature shard used to generate the dataset
 * @param minNumPartitions Minimum number of data partitions
 * @param numActiveDataPointsLowerBound The lower bound on the number of samples required to train a random effect model
 *                                      for an entity. If this bound is not met, the data is discarded.
 * @param numActiveDataPointsUpperBound The upper bound on the number of samples to keep (via reservoir sampling) as
 *                                      "active" for each individual-id level local dataset. The remaining samples that
 *                                      meet the numPassiveDataPointsToKeepLowerBound as discussed below will be kept as
 *                                      "passive" data.
 * @param numPassiveDataPointsLowerBound The lower bound on the number of data points required to create an
 *                                       individual-id level passive dataset using the data points leftover from the
 *                                       active dataset. In summary: IDs with fewer than
 *                                       [[numActiveDataPointsUpperBound]] samples will only an active dataset
 *                                       containing all samples; IDs with fewer than ([[numActiveDataPointsUpperBound]]
 *                                       + [[numPassiveDataPointsLowerBound]]) will have only an active dataset
 *                                       containing [[numActiveDataPointsUpperBound]] samples; all other IDs will have
 *                                       an active dataset containing [[numActiveDataPointsUpperBound]] samples and a
 *                                       passive dataset containing the remaining samples.
 * @param numFeaturesToSamplesRatioUpperBound The upper bound on the ratio between number of features and number of
 *                                            samples. Used for dimensionality reduction for IDs with very few samples.
 * @param projectorType The projector type, which is used to project the feature space of the random effect dataset
 *                      into a different space, usually one with lower dimension.
 */
case class RandomEffectDataConfiguration(
    randomEffectType: REType,
    featureShardId: FeatureShardId,
    minNumPartitions: Int = 1,
    numActiveDataPointsLowerBound: Option[Int] = None,
    numActiveDataPointsUpperBound: Option[Int] = None,
    numPassiveDataPointsLowerBound: Option[Int] = None,
    numFeaturesToSamplesRatioUpperBound: Option[Double] = None,
    projectorType: ProjectorType = IndexMapProjection)
  extends CoordinateDataConfiguration {

  require(
    numActiveDataPointsLowerBound.forall(_ > 0),
    s"Active data lower bound must be greater than 0: ${numActiveDataPointsLowerBound.get}")
  require(
    numActiveDataPointsUpperBound.forall(_ > 0),
    s"Active data upper bound must be greater than 0: ${numActiveDataPointsUpperBound.get}")
  require(
    numActiveDataPointsLowerBound.forall(_ <= numActiveDataPointsUpperBound.getOrElse(Int.MaxValue)),
    s"Active data lower bound must be less than active data upper bound (${numActiveDataPointsUpperBound.get})")
  require(
    numPassiveDataPointsLowerBound.forall(_ > 0),
    s"Passive data lower bound must be greater than 0: ${numPassiveDataPointsLowerBound.get}")
  require(
    numFeaturesToSamplesRatioUpperBound.forall(_ > 0),
    s"Features to samples ratio must be greater than 0: ${numFeaturesToSamplesRatioUpperBound.get}")
}
