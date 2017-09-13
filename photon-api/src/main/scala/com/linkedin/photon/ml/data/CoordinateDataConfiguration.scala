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
import com.linkedin.photon.ml.projector.ProjectorType

/**
 * Generic trait for a configuration to define a coordinate data set.
 */
protected[ml] sealed trait CoordinateDataConfiguration {

  def featureShardId: FeatureShardId
  def minNumPartitions: Int

  require(0 < minNumPartitions, s"Cannot have fewer than 1 Spark partition: $minNumPartitions")
}

/**
 * Configuration needed in order to generate a [[com.linkedin.photon.ml.data.FixedEffectDataSet]].
 *
 * @param featureShardId Key of the feature shard used to generate the data set
 * @param minNumPartitions Minimum number of data partitions
 */
case class FixedEffectDataConfiguration(featureShardId: FeatureShardId, minNumPartitions: Int)
  extends CoordinateDataConfiguration

/**
 * Configurations needed in order to generate a [[com.linkedin.photon.ml.data.RandomEffectDataSet]]
 *
 * @param randomEffectType The corresponding random effect type of the data set
 * @param featureShardId Key of the feature shard used to generate the data set
 * @param minNumPartitions Minimum number of data partitions
 * @param numActiveDataPointsUpperBound The upper bound on the number of samples to keep (via reservoir sampling) as
 *                                      "active" for each individual-id level local data set. The remaining samples that
 *                                      meet the numPassiveDataPointsToKeepLowerBound as discussed below will be kept as
 *                                      "passive" data.
 * @param numPassiveDataPointsLowerBound The lower bound on the number of data points required to create an
 *                                       individual-id level passive data set using the data points leftover from the
 *                                       active data set. In summary: IDs with fewer than
 *                                       [[numActiveDataPointsUpperBound]] samples will only an active data set
 *                                       containing all samples; IDs with fewer than ([[numActiveDataPointsUpperBound]]
 *                                       + [[numPassiveDataPointsLowerBound]]) will have only an active data set
 *                                       containing [[numActiveDataPointsUpperBound]] samples; all other IDs will have
 *                                       an active data set containing [[numActiveDataPointsUpperBound]] samples and a
 *                                       passive data set containing the remaining samples.
 * @param numFeaturesToSamplesRatioUpperBound The upper bound on the ratio between number of features and number of
 *                                            samples. Used for dimensionality reduction for IDs with very few samples.
 * @param projectorType The projector type, which is used to project the feature space of the random effect data set
 *                      into a different space, usually one with lower dimension.
 */
case class RandomEffectDataConfiguration(
    randomEffectType: REType,
    featureShardId: FeatureShardId,
    minNumPartitions: Int,
    numActiveDataPointsUpperBound: Option[Int],
    numPassiveDataPointsLowerBound: Option[Int],
    numFeaturesToSamplesRatioUpperBound: Option[Double],
    projectorType: ProjectorType)
  extends CoordinateDataConfiguration {

  require(
    numActiveDataPointsUpperBound.forall(_ > 0),
    s"Active data upper bound must be greater than 0: $numActiveDataPointsUpperBound")
  require(
    numPassiveDataPointsLowerBound.forall(_ > 0),
    s"Passive data lower bound must be greater than 0: $numPassiveDataPointsLowerBound")
  require(
    numFeaturesToSamplesRatioUpperBound.forall(_ > 0),
    s"Features to samples ratio must be greater than 0: $numFeaturesToSamplesRatioUpperBound")
}
