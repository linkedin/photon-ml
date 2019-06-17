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
package com.linkedin.photon.ml.io.scopt.avro

import org.apache.hadoop.fs.Path
import org.apache.spark.ml.param.ParamMap
import org.testng.Assert.assertEquals
import org.testng.annotations.Test

import com.linkedin.photon.ml.data.avro.NameAndTermFeatureBagsDriver
import com.linkedin.photon.ml.util.DateRange

/**
 * Unit tests for the [[ScoptNameAndTermFeatureBagsParametersParser]].
 */
class ScoptNameAndTermFeatureBagsParametersParserTest {

  /**
   * Test that a valid [[ParamMap]] can be roundtrip-ed by the parser (parameters -> string args -> parameters).
   */
  @Test
  def testRoundTrip(): Unit = {

    val inputPaths = Set(new Path("/some/input/path"))
    val inputDateRange = DateRange.fromDateString("20170101-20181231")
    val minInputPartitions = 11
    val outputPath = new Path("/some/output/path")
    val overrideOutputDir = true
    val featureBagKeys = Set("bag1", "bag2")
    val applicationName = "myApplication_name"

    val initialParamMap = ParamMap
      .empty
      .put(NameAndTermFeatureBagsDriver.inputDataDirectories, inputPaths)
      .put(NameAndTermFeatureBagsDriver.inputDataDateRange, inputDateRange)
      .put(NameAndTermFeatureBagsDriver.minInputPartitions, minInputPartitions)
      .put(NameAndTermFeatureBagsDriver.rootOutputDirectory, outputPath)
      .put(NameAndTermFeatureBagsDriver.overrideOutputDirectory, overrideOutputDir)
      .put(NameAndTermFeatureBagsDriver.featureBagsKeys, featureBagKeys)
      .put(NameAndTermFeatureBagsDriver.applicationName, applicationName)

    val finalParamMap = ScoptNameAndTermFeatureBagsParametersParser.parseFromCommandLine(
      ScoptNameAndTermFeatureBagsParametersParser.printForCommandLine(initialParamMap).flatMap(_.split(" ")).toArray)

    ScoptNameAndTermFeatureBagsParametersParser
      .scoptNameAndTermFeatureBagsParams
      .foreach { scoptParam =>
        assertEquals(finalParamMap.get(scoptParam.param), initialParamMap.get(scoptParam.param))
      }
  }
}
