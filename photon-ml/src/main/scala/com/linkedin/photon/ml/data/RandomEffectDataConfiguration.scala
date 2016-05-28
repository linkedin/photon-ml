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

import com.linkedin.photon.ml.projector.{IdentityProjection, IndexMapProjection, RandomProjection, ProjectorType}
import com.linkedin.photon.ml.projector.ProjectorType._

/**
 * Configurations needed in order to generate a [[RandomEffectDataSet]]
 *
 * @param randomEffectId The corresponding random effect Id of the data set
 * @param featureShardId Key of the feature shard used to generate the data set
 * @param numPartitions Number of partitions of the data
 * @param numActiveDataPointsToKeepUpperBound The upper bound on the number of samples to keep (via reservoir sampling)
 *                                           as "active" for each individual-id level local data set. The remaining
 *                                           samples that meets the numPassiveDataPointsToKeepLowerBound as discussed
 *                                           below will be kept as "passive" data.
 * @param numPassiveDataPointsToKeepLowerBound The lower bound on the number of passive data points to keep for each
 *                                           individual-id level local data set. Only those individual-id level local
 *                                           data set with number of passive data points larger than
 *                                           [[numPassiveDataPointsToKeepLowerBound]] will have its passive data kept
 *                                           during the processing step. Consequently, all the remaining passive data
 *                                           have more than [[numPassiveDataPointsToKeepLowerBound]] samples.
 * @param numFeaturesToSamplesRatioUpperBound The upper bound on the ratio between number of features and number of
 *                                            samples used for feature selection for each individual-id level local
 *                                            data set in the random effect data set.
 * @param projectorType The projector type, which is used to project the feature space of the random effect data set
 *                      into a different space, usually one with lower dimension.
 */
protected[ml] case class RandomEffectDataConfiguration private (
    randomEffectId: String,
    featureShardId: String,
    numPartitions: Int,
    numActiveDataPointsToKeepUpperBound: Int,
    numPassiveDataPointsToKeepLowerBound: Int,
    numFeaturesToSamplesRatioUpperBound: Double,
    projectorType: ProjectorType) {

  def isDownSamplingNeeded: Boolean  = numActiveDataPointsToKeepUpperBound < Int.MaxValue

  def isFeatureSelectionNeeded: Boolean = numFeaturesToSamplesRatioUpperBound < Double.MaxValue

  override def toString: String = {
    s"randomEffectId: $randomEffectId, featureShardId: $featureShardId, numPartitions: $numPartitions, " +
        s"numActiveDataPointsToKeepUpperBound: $numActiveDataPointsToKeepUpperBound, " +
        s"numPassiveDataPointsToKeepLowerBound: $numPassiveDataPointsToKeepLowerBound, " +
        s"numFeaturesToSamplesRatioUpperBound: $numFeaturesToSamplesRatioUpperBound, " +
        s"projectorType: $projectorType."
  }
}

object RandomEffectDataConfiguration {

  private val FIRST_LEVEL_SPLITTER = ","
  private val SECOND_LEVEL_SPLITTER = "="

  /**
   * Parse and build the [[RandomEffectDataConfiguration]] from the input [[String]]
   * @param string The input [[String]]
   * @return The parsed and built random effect data configuration
   */
  protected[ml] def parseAndBuildFromString(string: String): RandomEffectDataConfiguration = {

    val expectedTokenLength = 7
    val configParams = string.split(FIRST_LEVEL_SPLITTER)
    assert(configParams.length == expectedTokenLength, s"Cannot parse $string as random effect data configuration.\n" +
        s"The expected random effect data configuration should contain $expectedTokenLength parts separated by " +
        s"\'$FIRST_LEVEL_SPLITTER\'.")

    val randomEffectId = configParams(0)
    val featureShardKey = configParams(1)
    val numPartitions = configParams(2).toInt
    val rawUpperBoundNumActiveDataPointsToKeep = configParams(3).toInt
    val upperBoundNumActiveDataPointsToKeep = if (rawUpperBoundNumActiveDataPointsToKeep < 0) {
      Int.MaxValue
    } else {
      rawUpperBoundNumActiveDataPointsToKeep
    }
    val rawLowerBoundNumPassiveDataPointsToKeep = configParams(4).toInt
    val lowerBoundNumPassiveDataPointsToKeep = if (rawLowerBoundNumPassiveDataPointsToKeep < 0) {
      0
    } else {
      rawLowerBoundNumPassiveDataPointsToKeep
    }
    val rawUpperBoundNumFeaturesToSamplesRatio = configParams(5).toDouble
    val upperBoundNumFeaturesToSamplesRatio = if (rawUpperBoundNumFeaturesToSamplesRatio < 0) {
      Double.MaxValue
    } else {
      rawUpperBoundNumFeaturesToSamplesRatio
    }

    val projectorConfigParams = configParams(6).split(SECOND_LEVEL_SPLITTER)
    val projectorTypeName = ProjectorType.withName(projectorConfigParams.head.toUpperCase)
    val projectorType = projectorTypeName match {
      case RANDOM =>
        assert(projectorConfigParams.length == 2, s"If projector of type $RANDOM is selected, the projected space " +
            s"dimension needs to be specified. Correct configuration format is " +
            s"$RANDOM${SECOND_LEVEL_SPLITTER}projectedSpaceDimension.")
        val projectedSpaceDimension = projectorConfigParams.last.toInt
        RandomProjection(projectedSpaceDimension)
      case INDEX_MAP => IndexMapProjection
      case IDENTITY => IdentityProjection
      case _ => throw new UnsupportedOperationException(s"Unsupported projector name $projectorTypeName")
    }

    RandomEffectDataConfiguration(randomEffectId, featureShardKey, numPartitions, upperBoundNumActiveDataPointsToKeep,
      lowerBoundNumPassiveDataPointsToKeep, upperBoundNumFeaturesToSamplesRatio, projectorType)
  }
}
