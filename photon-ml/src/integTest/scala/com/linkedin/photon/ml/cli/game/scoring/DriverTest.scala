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
package com.linkedin.photon.ml.cli.game.scoring

import org.apache.hadoop.fs.Path
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.testng.Assert._
import org.testng.annotations.Test

import com.linkedin.photon.ml.avro.data.ScoreProcessingUtils
import com.linkedin.photon.ml.avro.{AvroUtils, DataProcessingUtilsTest}
import com.linkedin.photon.ml.constants.MathConst
import com.linkedin.photon.ml.test.{CommonTestUtils, SparkTestUtils, TestTemplateWithTmpDir}
import com.linkedin.photon.ml.util.{Utils, PhotonLogger}

class DriverTest extends SparkTestUtils with TestTemplateWithTmpDir {

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def failedTestRunWithOutputDirExists(): Unit = sparkTest("failedTestRunWithOutputDirExists") {
    val args = DriverTest.yahooMusicArgs(getTmpDir, deleteOutputDirIfExists = false)
    runDriver(CommonTestUtils.argArray(args))
  }

  @Test
  def endToEndRunWithYahooMusicDataSet(): Unit = sparkTest("endToEndRunWithYahooMusicDataSet") {
    val args = DriverTest.yahooMusicArgs(getTmpDir, deleteOutputDirIfExists = true)
    runDriver(CommonTestUtils.argArray(args))

    // Load the scores and compute the evaluation metric to see whether the scores make sense or not
    val scoreDir = new Path(args("output-dir"), Driver.SCORES).toString
    val uidAndScores = ScoreProcessingUtils.loadScoredItemsFromHDFS(scoreDir, sc)
        .map(scoredItem => (scoredItem.uid, scoredItem.predictionScore))
    val responseDirs = args("input-data-dirs").split(",")
    val responseRecords = AvroUtils.readAvroFiles(sc, responseDirs, minPartitions = 1)
    val uidAndResponses = DataProcessingUtilsTest.loadUidsWithResponsesFromGenericRecords(responseRecords)

    val predictionAndObservations = uidAndScores.rightOuterJoin(uidAndResponses).map { case (_, (score, response)) =>
      (score.getOrElse(0.0), response)
    }
    val rootMeanSquaredError = new RegressionMetrics(predictionAndObservations).rootMeanSquaredError

    // Compare with the RMSE capture from an assumed-correct implementation on 5/20/2016
    assertEquals(rootMeanSquaredError, 1.32106, MathConst.LOW_PRECISION_TOLERANCE_THRESHOLD)
  }

  /**
   * Run the Game driver with the specified arguments
   *
   * @param args the command-line arguments
   */
  private def runDriver(args: Array[String]): Driver = {
    val params = Params.parseFromCommandLine(args)
    val logger = new PhotonLogger(s"${params.outputDir}/log", sc)
    logger.setLogLevel(PhotonLogger.LogLevelDebug)
    val driver = new Driver(params, sc, logger)

    driver.run()
    logger.close()
    driver
  }
}

object DriverTest {

  /**
    * Arguments set for the Yahoo music data and model for the Game scoring driver
    */
  def yahooMusicArgs(outputDir: String, deleteOutputDirIfExists: Boolean): Map[String, String] = {
    val inputRoot = getClass.getClassLoader.getResource("GameIntegTest").getPath
    val inputDir = new Path(inputRoot, "input/test-with-uid").toString
    val featurePath = new Path(inputRoot, "input/feature-lists").toString
    val featureShardIdToFeatureSectionKeysMap =
      "globalShard:features,songFeatures,userFeatures|userShard:features,songFeatures|songShard:features,userFeatures"
    val randomEffectIdSet = "userId,songId"
    val modelDir = new Path(inputRoot, "gameModel").toString
    val numExecutors = "1"
    val applicationName = "GAME-Scoring-Integ-Test"
    Map(
      "input-data-dirs" -> inputDir,
      "feature-name-and-term-set-path" -> featurePath,
      "feature-shard-id-to-feature-section-keys-map" -> featureShardIdToFeatureSectionKeysMap,
      "random-effect-id-set" -> randomEffectIdSet,
      "game-model-input-dir" -> modelDir,
      "output-dir" -> outputDir,
      "num-files" -> numExecutors,
      "delete-output-dir-if-exists" -> deleteOutputDirIfExists.toString,
      "application-name" -> applicationName
    )
  }
}
