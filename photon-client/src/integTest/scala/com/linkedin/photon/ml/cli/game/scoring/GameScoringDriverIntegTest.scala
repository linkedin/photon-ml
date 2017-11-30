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

import org.apache.hadoop.fs.Path
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.testng.Assert._
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.data.avro.ScoreProcessingUtils
import com.linkedin.photon.ml.evaluation.EvaluatorType._
import com.linkedin.photon.ml.evaluation._
import com.linkedin.photon.ml.io.FeatureShardConfiguration
import com.linkedin.photon.ml.test.{CommonTestUtils, SparkTestUtils, TestTemplateWithTmpDir}
import com.linkedin.photon.ml.util.PhotonLogger

/**
 * Integration tests for [[GameScoringDriver]].
 */
class GameScoringDriverIntegTest extends SparkTestUtils with TestTemplateWithTmpDir {

  import GameScoringDriverIntegTest._

  /**
   * Test that a scoring job will fail when attempting to write to an existing directory.
   */
  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def failedTestRunWithOutputDirExists(): Unit = sparkTest("failedTestRunWithOutputDirExists") {

    val params = fixedEffectArgs
      .put(GameScoringDriver.rootOutputDirectory, new Path(getTmpDir))
      .put(GameScoringDriver.overrideOutputDirectory, false)

    runDriver(params)
  }

  /**
   * Test that the scoring job will correctly read/write the GAME model ID.
   */
  @Test
  def testGameModelId(): Unit = sparkTest("testGameModelId") {

    val modelId = "someModelIdForTest"
    val outputPath = new Path(getTmpDir)
    val scoresPath = new Path(outputPath, s"${GameScoringDriver.SCORES_DIR}")
    val params = fixedEffectArgs
      .put(GameScoringDriver.rootOutputDirectory, outputPath)
      .put(GameScoringDriver.modelId, modelId)

    runDriver(params)

    val loadedModelIdWithScoredItemAsRDD = ScoreProcessingUtils.loadScoredItemsFromHDFS(scoresPath.toString, sc)

    assertTrue(loadedModelIdWithScoredItemAsRDD.map(_._1).collect().forall(_ == modelId))
  }

  @DataProvider
  def numOutputFilesProvider(): Array[Array[Any]] = Array(
    Array(1, 1),
    Array(10, 3))

  /**
   * Test that the scoring job can correctly limit the maximum number of output files.
   *
   * @param outputFilesLimit The limit on the number of output files
   * @param expectedOutputFiles The expected number of output files
   */
  @Test(dataProvider = "numOutputFilesProvider")
  def testNumOutputFiles(outputFilesLimit: Int, expectedOutputFiles: Int): Unit = sparkTest("testNumOutputFiles") {

    val outputPath = new Path(getTmpDir)
    val scoresPath = new Path(outputPath, s"${GameScoringDriver.SCORES_DIR}")
    val params = fixedEffectArgs
      .put(GameScoringDriver.rootOutputDirectory, outputPath)
      .put(GameScoringDriver.outputFilesLimit, outputFilesLimit)

    runDriver(params)

    val fs = scoresPath.getFileSystem(sc.hadoopConfiguration)
    val numLoadedFiles = fs.listStatus(scoresPath).count(_.getPath.toString.contains("part"))

    assertEquals(numLoadedFiles, expectedOutputFiles)
  }

  /**
   * Test that the scoring job can score and evaluate correctly using a fixed effect model.
   */
  @Test
  def testEndToEndRunWithFixedEffectOnly(): Unit = sparkTest("testEndToEndRunWithFixedEffectOnly") {

    val outputPath = new Path(getTmpDir)
    val scoresPath = new Path(outputPath, s"${GameScoringDriver.SCORES_DIR}")
    val params = fixedEffectArgs
      .put(GameScoringDriver.rootOutputDirectory, outputPath)
      .put(GameScoringDriver.evaluators, Seq(RMSE))

    runDriver(params)

    // Load the scores and compute the evaluation metric to see whether the scores make sense or not
    val predictionAndObservations = ScoreProcessingUtils
      .loadScoredItemsFromHDFS(scoresPath.toString, sc)
      .map { case (_, scoredItem) => (scoredItem.predictionScore, scoredItem.label.get) }
    val rootMeanSquaredError = new RegressionMetrics(predictionAndObservations).rootMeanSquaredError

    // Compare with the RMSE capture from an assumed-correct implementation on 7/27/2016
    assertEquals(rootMeanSquaredError, 1.32171515, CommonTestUtils.LOW_PRECISION_TOLERANCE)
  }

  /**
   * Test that the scoring job can score and evaluate precision @ k correctly using a fixed effect model.
   */
//  @Test
  def testEvaluateFixedEffectOnlyWithPrecisionAtK(): Unit = sparkTest("testEvaluateFixedEffectOnlyWithPrecisionAtK") {

    val outputPath = new Path(getTmpDir)
    val userId = "userId"
    val songId = "songId"
    val numFeatures = "numFeatures"
    val params = fixedEffectArgs
      .put(GameScoringDriver.rootOutputDirectory, outputPath)
      .put(GameScoringDriver.evaluators,
        Seq(MultiPrecisionAtK(1, userId), MultiPrecisionAtK(1, songId), MultiPrecisionAtK(1, numFeatures)))

    // TODO: Need an actual check for something here
    runDriver(params)
  }

  /**
   * Test that the scoring job can score and evaluate correctly using a mixed effect model.
   */
  @Test
  def testEndToEndRunWithFullGLMix(): Unit = sparkTest("testEndToEndRunWithFullGLMix") {

    val outputPath = new Path(getTmpDir)
    val scoresPath = new Path(outputPath, s"${GameScoringDriver.SCORES_DIR}")
    val params = mixedEffectArgs
      .put(GameScoringDriver.rootOutputDirectory, outputPath)
      .put(GameScoringDriver.evaluators, Seq(RMSE))

    runDriver(params)

    // Load the scores and compute the evaluation metric to see whether the scores make sense or not
    val predictionAndObservations = ScoreProcessingUtils
      .loadScoredItemsFromHDFS(scoresPath.toString, sc)
      .map { case (_, scoredItem) => (scoredItem.predictionScore, scoredItem.label.get) }
    val rootMeanSquaredError = new RegressionMetrics(predictionAndObservations).rootMeanSquaredError

    // Compare with the RMSE capture from an assumed-correct implementation on 5/20/2016
    assertEquals(rootMeanSquaredError, 1.32106001, CommonTestUtils.LOW_PRECISION_TOLERANCE)
  }

  /**
   * Test that the scoring job can score and evaluate correctly, using an off-heap index map.
   */
  @Test
  def testOffHeapIndexMap(): Unit = sparkTest("testOffHeapIndexMap") {

    val outputPath = new Path(getTmpDir)
    val scoresPath = new Path(outputPath, s"${GameScoringDriver.SCORES_DIR}")
    val indexMapPath = new Path(
      getClass.getClassLoader.getResource("GameIntegTest/input/test-with-uid-feature-indexes").getPath)
    val params = mixedEffectArgs
      .put(GameScoringDriver.rootOutputDirectory, outputPath)
      .put(GameScoringDriver.offHeapIndexMapDirectory, indexMapPath)
      .put(GameScoringDriver.offHeapIndexMapPartitions, 1)
      .put(GameScoringDriver.evaluators, Seq(RMSE))
    params.remove(GameScoringDriver.featureBagsDirectory)

    runDriver(params)

    // Load the scores and compute the evaluation metric to see whether the scores make sense or not
    val predictionAndObservations = ScoreProcessingUtils
      .loadScoredItemsFromHDFS(scoresPath.toString, sc)
      .map { case (_, scoredItem) => (scoredItem.predictionScore, scoredItem.label.get) }
    val rootMeanSquaredError = new RegressionMetrics(predictionAndObservations).rootMeanSquaredError

    // Compare with the RMSE capture from an assumed-correct implementation on 5/20/2016
    assertEquals(rootMeanSquaredError, 1.32106001, CommonTestUtils.LOW_PRECISION_TOLERANCE)
  }

  /**
   * Run the GAME driver with the specified arguments.
   *
   * @param params Arguments for GAME scoring
   */
  def runDriver(params: ParamMap): Unit = {

    // Reset Driver parameters
    GameScoringDriver.clear()

    params.toSeq.foreach(paramPair => GameScoringDriver.set(paramPair.param.asInstanceOf[Param[Any]], paramPair.value))
    GameScoringDriver.sc = sc
    GameScoringDriver.logger = new PhotonLogger(
      new Path(GameScoringDriver.getOrDefault(GameScoringDriver.rootOutputDirectory), "log"),
      sc)
    GameScoringDriver.logger.setLogLevel(PhotonLogger.LogLevelDebug)
    GameScoringDriver.run()
    GameScoringDriver.logger.close()
  }
}

object GameScoringDriverIntegTest {

  private val inputRoot = getClass.getClassLoader.getResource("GameIntegTest").getPath
  private val inputPath = new Path(inputRoot, "input/test-with-uid")
  private val featurePath = new Path(inputRoot, "input/feature-lists")
  private val fixedEffectModelPath = new Path(inputRoot, "fixedEffectOnlyGAMEModel")
  private val mixedEffectModelPath = new Path(inputRoot, "gameModel")

  private val fixedEffectFeatureShardId = "globalShard"
  private val fixedEffectFeatureShardConfigs = Map(
    (fixedEffectFeatureShardId,
      FeatureShardConfiguration(Set("features", "songFeatures", "userFeatures"), hasIntercept = true)))

  private val perUserRandomEffectFeatureShardId = "userShard"
  private val perSongRandomEffectFeatureShardId = "songShard"
  private val mixedEffectFeatureShardConfigs = fixedEffectFeatureShardConfigs ++
    Map(
      (perUserRandomEffectFeatureShardId,
        FeatureShardConfiguration(Set("features", "songFeatures"), hasIntercept = true)),
      (perSongRandomEffectFeatureShardId,
        FeatureShardConfiguration(Set("features", "userFeatures"), hasIntercept = true)))

  /**
   * Default arguments to the GAME scoring driver.
   *
   * @return Arguments to run GAME scoring
   */
  private def defaultArgs: ParamMap =
    ParamMap
      .empty
      .put(GameScoringDriver.inputDataDirectories, Set(inputPath))
      .put(GameScoringDriver.featureBagsDirectory, featurePath)
      .put(GameScoringDriver.overrideOutputDirectory, true)

  /**
   * Fixed effect only arguments.
   *
   * @return Arguments to run GAME scoring
   */
  def fixedEffectArgs: ParamMap =
    defaultArgs
      .put(GameScoringDriver.featureShardConfigurations, fixedEffectFeatureShardConfigs)
      .put(GameScoringDriver.modelInputDirectory, fixedEffectModelPath)

  /**
   * Fixed and random effect arguments.
   *
   * @return Arguments to run GAME scoring
   */
  def mixedEffectArgs: ParamMap =
    defaultArgs
      .put(GameScoringDriver.featureShardConfigurations, mixedEffectFeatureShardConfigs)
      .put(GameScoringDriver.modelInputDirectory, mixedEffectModelPath)
}
