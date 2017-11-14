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
package com.linkedin.photon.ml.io.scopt.index

import org.apache.hadoop.fs.Path
import org.apache.spark.ml.param.ParamMap
import org.testng.Assert.assertEquals
import org.testng.annotations.Test

import com.linkedin.photon.ml.index.FeatureIndexingDriver
import com.linkedin.photon.ml.io.FeatureShardConfiguration
import com.linkedin.photon.ml.util.DateRange

/**
 * Unit tests for the [[ScoptFeatureIndexingParametersParser]].
 */
class ScoptFeatureIndexingParametersParserTest {

  /**
   * Test that a valid [[ParamMap]] can be roundtrip-ed by the parser (parameters -> string args -> parameters).
   */
  @Test
  def testRoundTrip(): Unit = {

    val inputPaths = Set(new Path("/some/input/path"))
    val inputDateRange = DateRange.fromDateString("20170101-20181231")
    val minInputPartitions = 1
    val outputPath = new Path("/some/output/path")
    val overrideOutputDir = true
    val applicationName = "myApplication_name"
    val numPartitions = 2

    val featureShard1 = "featureShard1"
    val featureBags1 = Set("bag1", "bag2")
    val featureShardIntercept1 = true
    val featureShardConfig1 = FeatureShardConfiguration(featureBags1, featureShardIntercept1)
    val featureShard2 = "featureShard2"
    val featureBags2 = Set("bag3", "bag4")
    val featureShardIntercept2 = false
    val featureShardConfig2 = FeatureShardConfiguration(featureBags2, featureShardIntercept2)
    val featureShardConfigs = Map(
      (featureShard1, featureShardConfig1),
      (featureShard2, featureShardConfig2))

    val initialParamMap = ParamMap
      .empty
      .put(FeatureIndexingDriver.inputDataDirectories, inputPaths)
      .put(FeatureIndexingDriver.inputDataDateRange, inputDateRange)
      .put(FeatureIndexingDriver.minInputPartitions, minInputPartitions)
      .put(FeatureIndexingDriver.rootOutputDirectory, outputPath)
      .put(FeatureIndexingDriver.overrideOutputDirectory, overrideOutputDir)
      .put(FeatureIndexingDriver.featureShardConfigurations, featureShardConfigs)
      .put(FeatureIndexingDriver.numPartitions, numPartitions)
      .put(FeatureIndexingDriver.applicationName, applicationName)

    val finalParamMap = ScoptFeatureIndexingParametersParser.parseFromCommandLine(
      ScoptFeatureIndexingParametersParser.printForCommandLine(initialParamMap).flatMap(_.split(" ")).toArray)

    ScoptFeatureIndexingParametersParser
      .scoptFeatureIndexingParams
      .foreach { scoptParam =>
        assertEquals(finalParamMap.get(scoptParam.param), initialParamMap.get(scoptParam.param))
      }
  }
}
