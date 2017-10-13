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
package com.linkedin.photon.ml.io.scopt

import org.apache.hadoop.fs.Path
import org.apache.spark.ml.param.ParamMap
import org.testng.Assert.assertEquals
import org.testng.annotations.Test

import com.linkedin.photon.ml.cli.game.scoring.GameScoringDriver
import com.linkedin.photon.ml.DataValidationType
import com.linkedin.photon.ml.data.InputColumnsNames
import com.linkedin.photon.ml.evaluation.EvaluatorType.{AUC, RMSE}
import com.linkedin.photon.ml.io.FeatureShardConfiguration
import com.linkedin.photon.ml.util.{DateRange, PhotonLogger}

/**
 * Unit tests for the [[ScoptGameScoringParametersParser]].
 */
class ScoptGameScoringParametersParserTest {

  /**
   * Test that a valid [[ParamMap]] can be roundtrip-ed by the parser (parameters -> string args -> parameters).
   */
  @Test
  def testRoundTrip(): Unit = {

    val inputPaths = Set(new Path("/some/input/path"))
    val inputDateRange = DateRange.fromDateString("20170101-20181231")
    val offHeapIndexMapPath = new Path("/some/off/heap/path")
    val offHeapIndexMapPartitions = 1
    val customColumnsNames = InputColumnsNames()
    InputColumnsNames.all.foreach(colName => customColumnsNames.updated(colName, s"___$colName"))
    val evaluators = Seq(AUC, RMSE)
    val outputPath = new Path("/some/output/path")
    val overrideOutputDir = true
    val featureBagsPath = new Path("/some/feature/bags/path")
    val dataValidation = DataValidationType.VALIDATE_SAMPLE
    val logLevel = PhotonLogger.parseLogLevelString("WARN")
    val applicationName = "myApplication_name"
    val outputFilesLimit = 3
    val modelInputDirectory = new Path("/some/model/path")
    val randomEffectTypes = Set("type1", "type2")
    val modelId = "someId"
    val logDataAndModelStats = true
    val spillScoresToDisk = true

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
      .put(GameScoringDriver.inputDataDirectories, inputPaths)
      .put(GameScoringDriver.inputDataDateRange, inputDateRange)
      .put(GameScoringDriver.offHeapIndexMapDirectory, offHeapIndexMapPath)
      .put(GameScoringDriver.offHeapIndexMapPartitions, offHeapIndexMapPartitions)
      .put(GameScoringDriver.inputColumnNames, customColumnsNames)
      .put(GameScoringDriver.evaluators, evaluators)
      .put(GameScoringDriver.rootOutputDirectory, outputPath)
      .put(GameScoringDriver.overrideOutputDirectory, overrideOutputDir)
      .put(GameScoringDriver.outputFilesLimit, outputFilesLimit)
      .put(GameScoringDriver.featureBagsDirectory, featureBagsPath)
      .put(GameScoringDriver.featureShardConfigurations, featureShardConfigs)
      .put(GameScoringDriver.dataValidation, dataValidation)
      .put(GameScoringDriver.logLevel, logLevel)
      .put(GameScoringDriver.applicationName, applicationName)
      .put(GameScoringDriver.modelInputDirectory, modelInputDirectory)
      .put(GameScoringDriver.randomEffectTypes, randomEffectTypes)
      .put(GameScoringDriver.modelId, modelId)
      .put(GameScoringDriver.logDataAndModelStats, logDataAndModelStats)
      .put(GameScoringDriver.spillScoresToDisk, spillScoresToDisk)

    val finalParamMap = ScoptGameScoringParametersParser.parseFromCommandLine(
      ScoptGameScoringParametersParser.printForCommandLine(initialParamMap).flatMap(_.split(" ")).toArray)

    ScoptGameScoringParametersParser
      .scoptGameScoringParams
      .foreach { scoptParam =>
        assertEquals(finalParamMap.get(scoptParam.param), initialParamMap.get(scoptParam.param))
      }
  }
}
