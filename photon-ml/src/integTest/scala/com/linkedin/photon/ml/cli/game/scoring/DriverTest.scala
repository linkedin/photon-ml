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

import scala.util.Random

import org.apache.hadoop.fs.Path
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.testng.Assert._
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.avro.data.ScoreProcessingUtils
import com.linkedin.photon.ml.constants.MathConst
import com.linkedin.photon.ml.data.{KeyValueScore, GameDatum}
import com.linkedin.photon.ml.evaluation._
import com.linkedin.photon.ml.test.{CommonTestUtils, SparkTestUtils, TestTemplateWithTmpDir}
import com.linkedin.photon.ml.util.PhotonLogger

class DriverTest extends SparkTestUtils with TestTemplateWithTmpDir {

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def failedTestRunWithUnsupportedEvaluatorType(): Unit = sparkTest("failedTestRunWithUnsupportedEvaluatorType") {
    val args = DriverTest.yahooMusicArgs(getTmpDir, evaluatorTypes = Seq("UnknownEvaluator"))
    runDriver(CommonTestUtils.argArray(args))
  }

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def failedTestRunWithNaNInGAMEData(): Unit = sparkTest("failedTestRunWithNaNInGAMEData") {
    val gameDatum = new GameDatum(response = Double.NaN, offset = 0.0, weight = 1.0, featureShardContainer = Map(),
      idTypeToValueMap = Map())
    val gameDataSet = sc.parallelize(Seq((1L, gameDatum)))
    val scores = new KeyValueScore(sc.parallelize(Seq((1L, 0.0))))
    Driver.evaluateScores(evaluatorType = SmoothedHingeLoss, scores = scores, gameDataSet)
  }

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def failedTestRunWithOutputDirExists(): Unit = sparkTest("failedTestRunWithOutputDirExists") {
    val args = DriverTest.yahooMusicArgs(getTmpDir, deleteOutputDirIfExists = false)
    runDriver(CommonTestUtils.argArray(args))
  }

  @Test
  def testGAMEModelId(): Unit = sparkTest("testGAMEModelId") {
    val modelId = "someModelIdForTest"
    val outputDir = getTmpDir
    val args = DriverTest.yahooMusicArgs(outputDir, fixedEffectOnly = true, modelId = modelId)
    runDriver(CommonTestUtils.argArray(args))

    val scoresDir = Driver.getScoresDir(outputDir)
    val loadedModelIdWithScoredItemAsRDD = ScoreProcessingUtils.loadScoredItemsFromHDFS(scoresDir, sc)
    assertTrue(loadedModelIdWithScoredItemAsRDD.map(_._1).collect().forall(_ == modelId))
  }

  @DataProvider
  def numOutputFilesProvider(): Array[Array[Any]] = {
    Array(Array(1), Array(10))
  }

  @Test(dataProvider = "numOutputFilesProvider")
  def testNumOutputFiles(numOutputFiles: Int): Unit = sparkTest("testNumOutputFiles") {
    val outputDir = getTmpDir
    val args = DriverTest.yahooMusicArgs(outputDir, numOutputFiles = numOutputFiles)
    runDriver(CommonTestUtils.argArray(args))
    val scoresPath = new Path(Driver.getScoresDir(outputDir))

    val fs = scoresPath.getFileSystem(sc.hadoopConfiguration)
    val numLoadedFiles = fs.listStatus(scoresPath).count(_.getPath.toString.contains("part"))
    assertEquals(numLoadedFiles, numOutputFiles)
  }

  @Test
  def endToEndRunWithFullGLMix(): Unit = sparkTest("endToEndRunWithFullGLMix") {
    val outputDir = getTmpDir
    val args = DriverTest.yahooMusicArgs(outputDir, fixedEffectOnly = false, deleteOutputDirIfExists = true)
    runDriver(CommonTestUtils.argArray(args))

    // Load the scores and compute the evaluation metric to see whether the scores make sense or not
    val scoreDir = Driver.getScoresDir(outputDir)
    val predictionAndObservations = ScoreProcessingUtils.loadScoredItemsFromHDFS(scoreDir, sc)
        .map { case (modelId, scoredItem) => (scoredItem.predictionScore, scoredItem.label.get) }

    val rootMeanSquaredError = new RegressionMetrics(predictionAndObservations).rootMeanSquaredError

    // Compare with the RMSE capture from an assumed-correct implementation on 5/20/2016
    assertEquals(rootMeanSquaredError, 1.32106, MathConst.LOW_PRECISION_TOLERANCE_THRESHOLD)
  }

  @Test
  def endToEndRunWithFixedEffectOnlyGLMix(): Unit = sparkTest("endToEndRunWithFixedEffectOnlyGLMix") {
    val outputDir = getTmpDir
    val args = DriverTest.yahooMusicArgs(outputDir, fixedEffectOnly = true, deleteOutputDirIfExists = true)
    runDriver(CommonTestUtils.argArray(args))

    // Load the scores and compute the evaluation metric to see whether the scores make sense or not
    val scoreDir = Driver.getScoresDir(outputDir)
    val predictionAndObservations = ScoreProcessingUtils.loadScoredItemsFromHDFS(scoreDir, sc)
        .map { case (modelId, scoredItem) => (scoredItem.predictionScore, scoredItem.label.get) }

    val rootMeanSquaredError = new RegressionMetrics(predictionAndObservations).rootMeanSquaredError

    // Compare with the RMSE capture from an assumed-correct implementation on 7/27/2016
    assertEquals(rootMeanSquaredError, 1.321715, MathConst.LOW_PRECISION_TOLERANCE_THRESHOLD)
  }

  @DataProvider
  def evaluatorTypeProvider(): Array[Array[Any]] = {
    Array(
      Array(Seq(AUC, LogisticLoss)),
      Array(Seq(PoissonLoss)),
      Array(Seq(RMSE, SquaredLoss)),
      Array(Seq(SmoothedHingeLoss)),
      Array(Seq(PrecisionAtK(1, "queryId"), PrecisionAtK(5, "documentId")))
    )
  }

  @Test(dataProvider = "evaluatorTypeProvider")
  def testEvaluateScores(evaluatorTypes: Seq[EvaluatorType]): Unit = sparkTest("testEvaluateScores") {
    val numSamples = 10
    val random = new Random(MathConst.RANDOM_SEED).self
    val scores = new KeyValueScore(sc.parallelize((0 until numSamples).map(idx => (idx.toLong, random.nextDouble()))))
    val labels = sc.parallelize((0 until numSamples).map(idx => (idx.toLong, random.nextInt(2))))
    val gameDataSet = labels.mapValues(label =>
      new GameDatum(
        response = label,
        offset = 0.0,
        weight = 1.0,
        featureShardContainer = Map(),
        idTypeToValueMap = Map("queryId" -> random.nextInt(2).toString, "documentId" -> random.nextInt(2).toString)
      )
    )
    evaluatorTypes.foreach { evaluatorType =>
      val computedMetric = Driver.evaluateScores(evaluatorType, scores, gameDataSet)
      val evaluator = Evaluator.buildEvaluator(evaluatorType, gameDataSet)
      val expectedMetric = evaluator.evaluate(scores.scores)
      assertEquals(computedMetric, expectedMetric, MathConst.MEDIUM_PRECISION_TOLERANCE_THRESHOLD)
    }
  }

  @Test
  def testOffHeapIndexMap(): Unit = sparkTest("offHeapIndexMap") {
    val outputDir = getTmpDir
    val indexMapPath = getClass.getClassLoader.getResource("GameIntegTest/input/test-with-uid-feature-indexes").getPath
    val args = DriverTest.yahooMusicArgs(outputDir, fixedEffectOnly = false, deleteOutputDirIfExists = true) ++ Map(
      "offheap-indexmap-dir" -> indexMapPath,
      "offheap-indexmap-num-partitions" -> "1")
    runDriver(CommonTestUtils.argArray(args))

    // Load the scores and compute the evaluation metric to see whether the scores make sense or not
    val scoreDir = Driver.getScoresDir(outputDir)
    val predictionAndObservations = ScoreProcessingUtils.loadScoredItemsFromHDFS(scoreDir, sc)
        .map { case (modelId, scoredItem) => (scoredItem.predictionScore, scoredItem.label.get) }

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
  def yahooMusicArgs(
      outputDir: String,
      fixedEffectOnly: Boolean = false,
      deleteOutputDirIfExists: Boolean = true,
      numOutputFiles: Int = 1,
      modelId: String = "",
      evaluatorTypes: Seq[String] = Seq("RMSE")): Map[String, String] = {

    val inputRoot = getClass.getClassLoader.getResource("GameIntegTest").getPath
    val inputDir = new Path(inputRoot, "input/test-with-uid").toString
    val featurePath = new Path(inputRoot, "input/feature-lists").toString

    val argumentsForGLMix =
      if (fixedEffectOnly) {
        val featureMap = "globalShard:features,songFeatures,userFeatures"
        val modelDir = new Path(inputRoot, "fixedEffectOnlyGAMEModel").toString
        Map("feature-shard-id-to-feature-section-keys-map" -> featureMap, "game-model-input-dir" -> modelDir)
      } else {
        val featureMap = "globalShard:features,songFeatures,userFeatures|userShard:features,songFeatures" +
            "|songShard:features,userFeatures"
        val idSet = "userId,songId"
        val modelDir = new Path(inputRoot, "gameModel").toString
        Map("feature-shard-id-to-feature-section-keys-map" -> featureMap, "game-model-input-dir" -> modelDir,
          "random-effect-id-set" -> idSet)
      }

    val applicationName = "GAME-Scoring-Integ-Test"

    Map(
      "input-data-dirs" -> inputDir,
      "feature-name-and-term-set-path" -> featurePath,
      "game-model-id" -> modelId,
      "output-dir" -> outputDir,
      "num-files" -> numOutputFiles.toString,
      "delete-output-dir-if-exists" -> deleteOutputDirIfExists.toString,
      "application-name" -> applicationName,
      "evaluator-type" -> evaluatorTypes.mkString(",")
    ) ++ argumentsForGLMix
  }
}
