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

/**
 * Configuration needed in order to generate a [[FixedEffectDataSet]].
 *
 * @param featureShardId Key of the feature shard used to generate the data set
 * @param minNumPartitions Minimum number of partitions of the fixed effect data
 */
protected[ml] case class FixedEffectDataConfiguration private (featureShardId: String, minNumPartitions: Int) {
  override def toString: String = s"featureShardId: $featureShardId, numPartitions: $minNumPartitions"
}

object FixedEffectDataConfiguration {

  protected[ml] val SPLITTER = ","
  protected[ml] val EXPECTED_FORMAT = s"featureShardId${SPLITTER}minNumPartitions"
  protected[ml] val EXPECTED_NUM_CONFIGS = 2

  /**
   * Parse and build the configuration object from a string representation.
   *
   * @param string The string representation
   * @return The configuration object
   */
  protected[ml] def parseAndBuildFromString(string: String): FixedEffectDataConfiguration = {

    val configParams = string.split(SPLITTER).map(_.trim)
    require(configParams.length == EXPECTED_NUM_CONFIGS,
      s"Parsing $string failed! The expected fixed effect data configuration should contain $EXPECTED_NUM_CONFIGS " +
          s"parts separated by \'$SPLITTER\', but found ${configParams.length}. Expected format: $EXPECTED_FORMAT")

    val featureShardId = configParams(0)
    val numPartitions = configParams(1).toInt
    FixedEffectDataConfiguration(featureShardId, numPartitions)
  }
}
