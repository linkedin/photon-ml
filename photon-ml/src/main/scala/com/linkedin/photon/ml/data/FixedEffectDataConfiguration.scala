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

/**
 * Configuration for a fixed effect dataset
 *
 * @author xazhang
 */
protected[ml] case class FixedEffectDataConfiguration(featureShardId: String, numPartitions: Int) {
  override def toString: String = s"featureShardId: $featureShardId, numPartitions: $numPartitions"
}

object FixedEffectDataConfiguration {

  private val SPLITTER = ","

  /**
   * Parse and build the configuration object from a string representation
   *
   * @param string the string representation
   * @return the configuration object
   */
  protected[ml] def parseAndBuildFromString(string: String): FixedEffectDataConfiguration = {

    val expectedTokenLength = 2
    val configParams = string.split(SPLITTER)
    assert(configParams.length == expectedTokenLength, s"Cannot parse $string as fixed effect data configuration.\n" +
        s"The expected fixed effect data configuration should contain $expectedTokenLength parts separated by " +
        s"\'$SPLITTER\'.")

    val featureShardId = configParams(0)
    val numPartitions = configParams(1).toInt
    FixedEffectDataConfiguration(featureShardId, numPartitions)
  }
}
