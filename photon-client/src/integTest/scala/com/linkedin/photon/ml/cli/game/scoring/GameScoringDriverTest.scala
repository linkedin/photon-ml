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
package com.linkedin.photon.ml.cli.game.scoring

import scala.util.Random

import org.apache.hadoop.fs.Path
import org.apache.spark.SparkException
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.testng.Assert._
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.constants.MathConst
import com.linkedin.photon.ml.data.GameDatum
import com.linkedin.photon.ml.data.avro.ScoreProcessingUtils
import com.linkedin.photon.ml.data.scoring.ModelDataScores
import com.linkedin.photon.ml.evaluation.EvaluatorType._
import com.linkedin.photon.ml.evaluation._
import com.linkedin.photon.ml.test.{CommonTestUtils, SparkTestUtils, TestTemplateWithTmpDir}
import com.linkedin.photon.ml.util.PhotonLogger

/**
 * Integration tests for [[GameScoringDriver]].
 */
class GameScoringDriverTest extends SparkTestUtils with TestTemplateWithTmpDir {

  import GameScoringDriverTest._

  /**
   * Test that a scoring job using an unknown evaluator will fail.
   */
  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def failedTestRunWithUnsupportedEvaluatorType(): Unit = sparkTest("failedTestRunWithUnsupportedEvaluatorType") {

    val args = yahooMusicArgs(getTmpDir, evaluatorTypes = Seq("UnknownEvaluator"))
    runDriver(CommonTestUtils.argArray(args))
  }

  /**
   * Test that a scoring job run on invalid data will fail.
   */
  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def failedTestRunWithNaNInGameData(): Unit = sparkTest("failedTestRunWithNaNInGameData") {

    val gameDatum = new GameDatum(
      response = Double.NaN,
      offsetOpt = Some(0.0),
      weightOpt = Some(1.0),
      featureShardContainer = Map(),
      idTagToValueMap = Map())
    val scoredDatum = gameDatum.toScoredGameDatum()
    val gameDataSet = sc.parallelize(Seq((1L, gameDatum)))
    val scores = new ModelDataScores(sc.parallelize(Seq((1L, scoredDatum))))
    GameScoringDriver.evaluateScores(evaluatorType = SmoothedHingeLoss, scores = scores, gameDataSet)
  }

  /**
   * Test that a scoring job will fail when attempting to write to an existing directory.
   */
  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def failedTestRunWithOutputDirExists(): Unit = sparkTest("failedTestRunWithOutputDirExists") {

    val args = yahooMusicArgs(getTmpDir, deleteOutputDirIfExists = false)
    runDriver(CommonTestUtils.argArray(args))
  }

  /**
   * Test that the scoring job will correctly read/write the GAME model ID.
   */
  @Test
  def testGameModelId(): Unit = sparkTest("testGameModelId") {

    val modelId = "someModelIdForTest"
    val outputDir = getTmpDir
    val args = yahooMusicArgs(outputDir, fixedEffectOnly = true, modelId = modelId)
    runDriver(CommonTestUtils.argArray(args))

    val scoreDir = s"$outputDir/${GameScoringDriver.SCORES_DIR}"
    val loadedModelIdWithScoredItemAsRDD = ScoreProcessingUtils.loadScoredItemsFromHDFS(scoreDir, sc)
    assertTrue(loadedModelIdWithScoredItemAsRDD.map(_._1).collect().forall(_ == modelId))
  }

  @DataProvider
  def numOutputFilesProvider(): Array[Array[Any]] = {
    Array(Array(1), Array(10))
  }

  /**
   * Test that the scoring job can correctly limit the maximum number of output files.
   *
   * @param numOutputFiles The limit on the number of output files
   */
  @Test(dataProvider = "numOutputFilesProvider")
  def testNumOutputFiles(numOutputFiles: Int): Unit = sparkTest("testNumOutputFiles") {

    val outputDir = getTmpDir
    val args = yahooMusicArgs(outputDir, numOutputFiles = numOutputFiles)
    runDriver(CommonTestUtils.argArray(args))
    val scoresPath = new Path(s"$outputDir/${GameScoringDriver.SCORES_DIR}")

    val fs = scoresPath.getFileSystem(sc.hadoopConfiguration)
    val numLoadedFiles = fs.listStatus(scoresPath).count(_.getPath.toString.contains("part"))
    assertEquals(numLoadedFiles, numOutputFiles)
  }

  /**
   * Test that the scoring job can score and evaluate correctly using a mixed effect model.
   */
  @Test
  def tesEendToEndRunWithFullGLMix(): Unit = sparkTest("endToEndRunWithFullGLMix") {

    val outputDir = getTmpDir
    val args = yahooMusicArgs(outputDir)
    runDriver(CommonTestUtils.argArray(args))

    // Load the scores and compute the evaluation metric to see whether the scores make sense or not
    val scoreDir = s"$outputDir/${GameScoringDriver.SCORES_DIR}"
    val predictionAndObservations = ScoreProcessingUtils.loadScoredItemsFromHDFS(scoreDir, sc)
        .map { case (_, scoredItem) => (scoredItem.predictionScore, scoredItem.label.get) }

    val rootMeanSquaredError = new RegressionMetrics(predictionAndObservations).rootMeanSquaredError

    // Compare with the RMSE capture from an assumed-correct implementation on 5/20/2016
    assertEquals(rootMeanSquaredError, 1.32106001, CommonTestUtils.LOW_PRECISION_TOLERANCE)
  }

  /**
   * Test that the scoring job can score and evaluate correctly using a fixed effect model.
   */
  @Test
  def testEndToEndRunWithFixedEffectOnlyGLMix(): Unit = sparkTest("endToEndRunWithFixedEffectOnlyGLMix") {

    val outputDir = getTmpDir
    val args = yahooMusicArgs(outputDir, fixedEffectOnly = true)

    runDriver(CommonTestUtils.argArray(args))

    // Load the scores and compute the evaluation metric to see whether the scores make sense or not
    val scoreDir = s"$outputDir/${GameScoringDriver.SCORES_DIR}"
    val predictionAndObservations = ScoreProcessingUtils.loadScoredItemsFromHDFS(scoreDir, sc)
        .map { case (_, scoredItem) => (scoredItem.predictionScore, scoredItem.label.get) }

    val rootMeanSquaredError = new RegressionMetrics(predictionAndObservations).rootMeanSquaredError

    // Compare with the RMSE capture from an assumed-correct implementation on 7/27/2016
    assertEquals(rootMeanSquaredError, 1.32171515, CommonTestUtils.LOW_PRECISION_TOLERANCE)
  }

  /**
   * Test that the scoring job can score and evaluate precision @ k correctly using a fixed effect model.
   */
//  @Test
  def testEvaluateFixedEffectOnlyGLMixWithPrecisionAtK(): Unit =
    sparkTest("evaluateFixedEffectOnlyGLMixWithPrecisionAtK") {

      val args = yahooMusicArgs(
        getTmpDir,
        fixedEffectOnly = true,
        evaluatorTypes = Seq("precision@1:userId, precision@5:songId, precision@10:numFeatures"))

      // TODO: Need an actual check for something here
      runDriver(CommonTestUtils.argArray(args))
    }

  /**
   * Test that the scoring job will fail to evaluate precision @ k on an unknown ID.
   */
  @Test(expectedExceptions = Array(classOf[SparkException]))
  def testEvaluateFullModelWithPrecisionAtKOfUnknownId(): Unit =
    sparkTest("evaluateFullModelWithPrecisionAtKOfUnknownId") {
      val args = yahooMusicArgs(
        getTmpDir,
        evaluatorTypes = Seq("precision@1:userId, precision@5:foo, precision@10:bar"))

      runDriver(CommonTestUtils.argArray(args))
  }

  @DataProvider
  def evaluatorTypeProvider(): Array[Array[Any]] = Array(
      Array(Seq(AUC, LogisticLoss)),
      Array(Seq(PoissonLoss)),
      Array(Seq(RMSE, SquaredLoss)),
      Array(Seq(SmoothedHingeLoss)),
      Array(Seq(MultiPrecisionAtK(1, "queryId"), MultiPrecisionAtK(5, "documentId"))),
      Array(Seq(MultiAUC("queryId"), MultiAUC("documentId"))))

  /**
   * Test that the scoring job can correctly evaluate scores for multiple evaluators.
   *
   * @param evaluatorTypes The evaluators to use for score evaluation
   */
  @Test(dataProvider = "evaluatorTypeProvider")
  def testEvaluateScores(evaluatorTypes: Seq[EvaluatorType]): Unit = sparkTest("testEvaluateScores") {
    val numSamples = 10
    val random = new Random(MathConst.RANDOM_SEED).self
    val labels = sc.parallelize((0 until numSamples).map(idx => (idx.toLong, random.nextInt(2))))

    // Ensure that each queryId and documentId has both positive and negative labels, so that AUC will not return NaN
    val staticGameData = Seq((0, 1), (1, 0)).flatMap { case (queryId, documentId) =>
      Seq(0, 1).map { label =>
        new GameDatum(
          response = label,
          offsetOpt = Some(0.0),
          weightOpt = Some(1.0),
          featureShardContainer = Map(),
          idTagToValueMap = Map(("queryId", queryId.toString), ("documentId", documentId.toString))
        )
      }
    }

    // Add ids and parallelize
    val staticGameDataSet = sc.parallelize(
      (numSamples until numSamples + staticGameData.length)
        .map(_.toLong)
        .zip(staticGameData))

    // Union static data with generated data
    val gameDataSet = sc.union(staticGameDataSet, labels.mapValues(label =>
      new GameDatum(
        response = label,
        offsetOpt = Some(0.0),
        weightOpt = Some(1.0),
        featureShardContainer = Map(),
        idTagToValueMap = Map(("queryId", random.nextInt(2).toString), ("documentId", random.nextInt(2).toString))
      )
    ))

    val scores = new ModelDataScores(gameDataSet.mapValues(datum => datum.toScoredGameDatum(random.nextDouble())))

    evaluatorTypes.foreach { evaluatorType =>
      val computedMetric = GameScoringDriver.evaluateScores(evaluatorType, scores, gameDataSet)
      val evaluator = EvaluatorFactory.buildEvaluator(evaluatorType, gameDataSet)
      val expectedMetric = evaluator.evaluate(scores.scores.mapValues(_.score))
      assertEquals(computedMetric, expectedMetric, CommonTestUtils.HIGH_PRECISION_TOLERANCE)
    }
  }

  /**
   * Test that the scoring job can score and evaluate correctly, using an off-heap index map.
   */
  @Test
  def testOffHeapIndexMap(): Unit = sparkTest("offHeapIndexMap") {
    val outputDir = getTmpDir
    val indexMapPath = getClass.getClassLoader.getResource("GameIntegTest/input/test-with-uid-feature-indexes").getPath
    val args = yahooMusicArgs(outputDir) ++ Map(
      ("offheap-indexmap-dir", indexMapPath),
      ("offheap-indexmap-num-partitions", "1"))

    runDriver(CommonTestUtils.argArray(args))

    // Load the scores and compute the evaluation metric to see whether the scores make sense or not
    val scoreDir = s"$outputDir/${GameScoringDriver.SCORES_DIR}"
    val predictionAndObservations = ScoreProcessingUtils.loadScoredItemsFromHDFS(scoreDir, sc)
        .map { case (_, scoredItem) => (scoredItem.predictionScore, scoredItem.label.get) }

    val rootMeanSquaredError = new RegressionMetrics(predictionAndObservations).rootMeanSquaredError

    // Compare with the RMSE capture from an assumed-correct implementation on 5/20/2016
    assertEquals(rootMeanSquaredError, 1.32106001, CommonTestUtils.LOW_PRECISION_TOLERANCE)
  }

  /**
   * Run the GAME driver with the specified arguments.
   *
   * @param args The command-line arguments
   */
  private def runDriver(args: Array[String]): Unit = {

    GameScoringDriver.sc = sc
    GameScoringDriver.parameters = GameScoringParams.parseFromCommandLine(args)
    GameScoringDriver.logger = new PhotonLogger(
      new Path(GameScoringDriver.parameters.outputDir, "log"),
      sc)
    GameScoringDriver.logger.setLogLevel(PhotonLogger.LogLevelDebug)
    GameScoringDriver.run()
    GameScoringDriver.logger.close()
  }
}

object GameScoringDriverTest {

  /**
   * Arguments set for the Yahoo music data and model for the GAME scoring driver.
   *
   * @param outputDir The output directory when running Game
   * @param fixedEffectOnly Whether to use fixed effects only
   * @param deleteOutputDirIfExists Whether to delete the output diretory or not
   * @param numOutputFiles The number of output files to use
   * @param modelId The model id
   * @param evaluatorTypes The types of evaluators to use
   * @return A well-formed set of arguments to run Yahoo music
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
      ("input-data-dirs", inputDir),
      ("feature-name-and-term-set-path", featurePath),
      ("game-model-id", modelId),
      ("output-dir", outputDir),
      ("num-files", numOutputFiles.toString),
      ("delete-output-dir-if-exists", deleteOutputDirIfExists.toString),
      ("application-name", applicationName),
      ("evaluator-type", evaluatorTypes.mkString(","))
    ) ++ argumentsForGLMix
  }
}
