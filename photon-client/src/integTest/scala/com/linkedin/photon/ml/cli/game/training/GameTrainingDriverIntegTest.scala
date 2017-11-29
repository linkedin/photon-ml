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
package com.linkedin.photon.ml.cli.game.training

import java.io.File

import scala.collection.JavaConversions._

import org.apache.hadoop.fs.Path
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.param.{Param, ParamMap}
import org.testng.Assert._
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.avro.generated.BayesianLinearModelAvro
import com.linkedin.photon.ml.{DataValidationType, TaskType}
import com.linkedin.photon.ml.cli.game.GameDriver
import com.linkedin.photon.ml.constants.StorageLevel
import com.linkedin.photon.ml.data.{FixedEffectDataConfiguration, GameConverters, RandomEffectDataConfiguration}
import com.linkedin.photon.ml.data.avro._
import com.linkedin.photon.ml.estimators.GameEstimator
import com.linkedin.photon.ml.evaluation.RMSEEvaluator
import com.linkedin.photon.ml.io.{FeatureShardConfiguration, FixedEffectCoordinateConfiguration, ModelOutputMode, RandomEffectCoordinateConfiguration}
import com.linkedin.photon.ml.optimization._
import com.linkedin.photon.ml.optimization.game.{FixedEffectOptimizationConfiguration, RandomEffectOptimizationConfiguration}
import com.linkedin.photon.ml.projector.{IndexMapProjection, RandomProjection}
import com.linkedin.photon.ml.test.{SparkTestUtils, TestTemplateWithTmpDir}
import com.linkedin.photon.ml.util._

/**
 * Test cases for the GAME training driver.
 *
 * Most of the cases are based on the Yahoo! Music data set in:
 * photon-ml/photon-client/src/integTest/resources/GameIntegTest/input/train/yahoo-music-train.avro
 */
class GameTrainingDriverIntegTest extends SparkTestUtils with GameTestUtils with TestTemplateWithTmpDir {

  import GameTrainingDriverIntegTest._

  Logger.getLogger("org").setLevel(Level.OFF) // turn off noisy Spark log

  /**
   * Test that GAME training will fail if the output directory already exists.
   */
  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def failedTestRunWithOutputDirExists(): Unit = sparkTest("failedTestRunWithOutputDirExists") {

    val outputDir = new Path(getTmpDir, "failedTestRunWithOutputDirExists")
    Utils.createHDFSDir(outputDir, sc.hadoopConfiguration)

    runDriver(fixedEffectToyRunArgs.put(GameTrainingDriver.rootOutputDirectory, outputDir))
  }

  /**
   * Test GAME training with a fixed effect model only, and an intercept.
   *
   * @note Intercepts are optional in [[GameEstimator]], but [[GameDriver]] will setup an intercept by default. This
   *       happens in [[GameDriver.prepareFeatureMapsDefault()]], and there only.
   */
  @Test
  def testFixedEffectsWithIntercept(): Unit = sparkTest("testFixedEffectsWithIntercept", useKryo = true) {

    // This is a baseline RMSE capture from an assumed-correct implementation on 4/14/2016
    val errorThreshold = 1.7
    val outputDir = new Path(getTmpDir, "fixedEffects")

    runDriver(fixedEffectSeriousRunArgs.put(GameTrainingDriver.rootOutputDirectory, outputDir))

    val allFixedEffectModelPath = allModelPath(outputDir, "fixed-effect", fixedEffectCoordinateId)
    val bestFixedEffectModelPath = bestModelPath(outputDir, "fixed-effect", fixedEffectCoordinateId)
    val fs = outputDir.getFileSystem(sc.hadoopConfiguration)

    assertTrue(fs.exists(allFixedEffectModelPath))
    assertTrue(fs.exists(bestFixedEffectModelPath))

    assertModelSane(allFixedEffectModelPath, expectedNumCoefficients = 14983)
    assertModelSane(bestFixedEffectModelPath, expectedNumCoefficients = 14983)

    assertTrue(evaluateModel(new Path(outputDir, "models/0")) < errorThreshold)
    assertTrue(evaluateModel(new Path(outputDir, "best")) < errorThreshold)

    assertTrue(AvroUtils.modelContainsIntercept(sc, allFixedEffectModelPath))
    assertTrue(AvroUtils.modelContainsIntercept(sc, bestFixedEffectModelPath))
  }

  /**
   * Test GAME training with a fixed effect model only, and an intercept, and no validation.
   *
   * @note Intercepts are optional in [[GameEstimator]], but [[GameDriver]] will setup an intercept by default. This
   *       happens in [[GameDriver.prepareFeatureMapsDefault()]], and there only.
   */
  @Test
  def testFixedEffectsWithAdditionalOpts(): Unit = sparkTest("testFixedEffectsWithIntercept", useKryo = true) {

    // This is a baseline RMSE capture from an assumed-correct implementation on 4/14/2016
    val errorThreshold = 1.7
    val outputDir = new Path(getTmpDir, "fixedEffects")
    val newArgs = fixedEffectSeriousRunArgs
      .copy
      .put(GameTrainingDriver.rootOutputDirectory, outputDir)
      .put(GameTrainingDriver.overrideOutputDirectory, true)
    newArgs.remove(GameTrainingDriver.validationDataDirectories)

    Utils.createHDFSDir(outputDir, sc.hadoopConfiguration)
    runDriver(newArgs)

    val allFixedEffectModelPath = allModelPath(outputDir, "fixed-effect", fixedEffectCoordinateId)
    val bestFixedEffectModelPath = bestModelPath(outputDir, "fixed-effect", fixedEffectCoordinateId)
    val fs = outputDir.getFileSystem(sc.hadoopConfiguration)

    assertTrue(fs.exists(allFixedEffectModelPath))
    assertFalse(fs.exists(bestFixedEffectModelPath))

    assertModelSane(allFixedEffectModelPath, expectedNumCoefficients = 14983)
    assertTrue(evaluateModel(new Path(outputDir, "models/0")) < errorThreshold)
    assertTrue(AvroUtils.modelContainsIntercept(sc, allFixedEffectModelPath))
  }

  /**
   * Test GAME training with a fixed effect model only, and no intercept.
   *
   * @note Since intercept terms are ON by default, they need to be explicitly disabled for this test.
   */
  @Test
  def testFixedEffectsWithoutIntercept(): Unit = sparkTest("testFixedEffectsWithoutIntercept", useKryo = true) {

    // This is a baseline RMSE capture from an assumed-correct implementation on 4/14/2016
    val errorThreshold = 1.7
    val outputDir = new Path(getTmpDir, "fixedEffects")
    val modifiedFeatureShardConfigs = fixedEffectFeatureShardConfigs
      .mapValues(_.copy(hasIntercept = false))
      .map(identity)

    runDriver(
      fixedEffectSeriousRunArgs
        .put(GameTrainingDriver.rootOutputDirectory, outputDir)
        .put(GameTrainingDriver.featureShardConfigurations, modifiedFeatureShardConfigs))

    val allFixedEffectModelPath = allModelPath(outputDir, "fixed-effect", fixedEffectCoordinateId)
    val bestFixedEffectModelPath = bestModelPath(outputDir, "fixed-effect", fixedEffectCoordinateId)
    val fs = outputDir.getFileSystem(sc.hadoopConfiguration)

    assertTrue(fs.exists(allFixedEffectModelPath))
    assertTrue(fs.exists(bestFixedEffectModelPath))

    assertModelSane(allFixedEffectModelPath, expectedNumCoefficients = 14980)
    assertModelSane(bestFixedEffectModelPath, expectedNumCoefficients = 14980)

    assertTrue(evaluateModel(new Path(outputDir, "models/0")) < errorThreshold)
    assertTrue(evaluateModel(new Path(outputDir, "best")) < errorThreshold)

    assertFalse(AvroUtils.modelContainsIntercept(sc, allFixedEffectModelPath))
    assertFalse(AvroUtils.modelContainsIntercept(sc, bestFixedEffectModelPath))
  }

  /**
   * Check that it's possible to train an intercept-only model.
   */
  @Test
  def testFixedEffectInterceptOnly(): Unit = sparkTest("testFixedEffectInterceptOnly", useKryo = true) {

    val outputDir = new Path(getTmpDir, "fixedEffectsInterceptOnly")
    val modifiedFeatureShardConfigs = fixedEffectFeatureShardConfigs
      .mapValues(_.copy(featureBags = Set()))
      .map(identity)
    val params = fixedEffectSeriousRunArgs
      .put(GameTrainingDriver.rootOutputDirectory, outputDir)
      .put(GameTrainingDriver.featureShardConfigurations, modifiedFeatureShardConfigs)
    params.remove(GameTrainingDriver.featureBagsDirectory)

    runDriver(params)

    val allFixedEffectModelPath = allModelPath(outputDir, "fixed-effect", fixedEffectCoordinateId)
    val bestFixedEffectModelPath = bestModelPath(outputDir, "fixed-effect", fixedEffectCoordinateId)
    val fs = outputDir.getFileSystem(sc.hadoopConfiguration)

    assertTrue(fs.exists(allFixedEffectModelPath))
    assertTrue(fs.exists(bestFixedEffectModelPath))

    assertModelSane(allFixedEffectModelPath, expectedNumCoefficients = 1)
    assertModelSane(bestFixedEffectModelPath, expectedNumCoefficients = 1)

    assertTrue(AvroUtils.modelContainsIntercept(sc, allFixedEffectModelPath))
    assertTrue(AvroUtils.modelContainsIntercept(sc, bestFixedEffectModelPath))
  }

  /**
   * Check that it's possible to train an intercept-only model with feature whitelists.
   */
  @Test
  def testFixedEffectInterceptOnlyFeatureBagsDir(): Unit = sparkTest("testFixedEffectInterceptOnly", useKryo = true) {

    val outputDir = new Path(getTmpDir, "fixedEffectsInterceptOnly")
    val modifiedFeatureShardConfigs = fixedEffectFeatureShardConfigs
      .mapValues(_.copy(featureBags = Set()))
      .map(identity)

    runDriver(
      fixedEffectSeriousRunArgs
        .put(GameTrainingDriver.rootOutputDirectory, outputDir)
        .put(GameTrainingDriver.featureShardConfigurations, modifiedFeatureShardConfigs))

    val allFixedEffectModelPath = allModelPath(outputDir, "fixed-effect", fixedEffectCoordinateId)
    val bestFixedEffectModelPath = bestModelPath(outputDir, "fixed-effect", fixedEffectCoordinateId)
    val fs = outputDir.getFileSystem(sc.hadoopConfiguration)

    assertTrue(fs.exists(allFixedEffectModelPath))
    assertTrue(fs.exists(bestFixedEffectModelPath))

    assertModelSane(allFixedEffectModelPath, expectedNumCoefficients = 1)
    assertModelSane(bestFixedEffectModelPath, expectedNumCoefficients = 1)

    assertTrue(AvroUtils.modelContainsIntercept(sc, allFixedEffectModelPath))
    assertTrue(AvroUtils.modelContainsIntercept(sc, bestFixedEffectModelPath))
  }

  /**
   * Test GAME training with a random effect models only, and intercepts.
   */
  @Test
  def testRandomEffectsWithIntercept(): Unit = sparkTest("testRandomEffectsWithIntercept", useKryo = true) {

    // This is a baseline RMSE capture from an assumed-correct implementation on 4/14/2016
    val errorThreshold = 2.34
    val outputDir = new Path(getTmpDir, "randomEffects")

    runDriver(randomEffectSeriousRunArgs.put(GameTrainingDriver.rootOutputDirectory, outputDir))

    val modelPaths = randomEffectCoordinateIds.map(bestModelPath(outputDir, "random-effect", _))
    val fs = outputDir.getFileSystem(sc.hadoopConfiguration)

    modelPaths.foreach { path =>
      assertTrue(fs.exists(path))
      assertModelSane(path, expectedNumCoefficients = 21)
      assertTrue(AvroUtils.modelContainsIntercept(sc, path))
    }

    assertTrue(evaluateModel(new Path(outputDir, "best")) < errorThreshold)
  }

  /**
   * Test GAME training with a random effect models only, and no intercepts.
   */
  @Test
  def testRandomEffectsWithoutAnyIntercept(): Unit = sparkTest("testRandomEffectsWithoutAnyIntercept", useKryo = true) {

    // This is a baseline RMSE capture from an assumed-correct implementation on 4/14/2016
    val errorThreshold = 2.34
    val outputDir = new Path(getTmpDir, "randomEffects")
    val modifiedFeatureShardConfigs = randomEffectFeatureShardConfigs
      .mapValues(_.copy(hasIntercept = false))
      .map(identity)

    runDriver(
      randomEffectSeriousRunArgs
        .put(GameTrainingDriver.rootOutputDirectory, outputDir)
        .put(GameTrainingDriver.featureShardConfigurations, modifiedFeatureShardConfigs))

    val modelPaths = randomEffectCoordinateIds.map(bestModelPath(outputDir, "random-effect", _))
    val fs = outputDir.getFileSystem(sc.hadoopConfiguration)

    modelPaths.foreach { path =>
      assertTrue(fs.exists(path))
      assertModelSane(path, expectedNumCoefficients = 20)
      assertFalse(AvroUtils.modelContainsIntercept(sc, path))
    }

    assertTrue(evaluateModel(new Path(outputDir, "best")) < errorThreshold)
  }

  /**
   * Test GAME training with a both fixed and random effect models.
   */
  @Test
  def testFixedAndRandomEffects(): Unit = sparkTest("fixedAndRandomEffects", useKryo = true) {

    // This is a baseline RMSE capture from an assumed-correct implementation on 4/14/2016
    val errorThreshold = 2.2
    val outputDir = new Path(getTmpDir, "fixedAndRandomEffects")

    runDriver(mixedEffectSeriousRunArgs.put(GameTrainingDriver.rootOutputDirectory, outputDir))

    val globalModelPath = bestModelPath(outputDir, "fixed-effect", "global")
    val userModelPath = bestModelPath(outputDir, "random-effect", "per-user")
    val songModelPath = bestModelPath(outputDir, "random-effect", "per-song")
    val artistModelPath = bestModelPath(outputDir, "random-effect", "per-artist")
    val fs = outputDir.getFileSystem(sc.hadoopConfiguration)

    assertTrue(fs.exists(globalModelPath))
    assertModelSane(globalModelPath, expectedNumCoefficients = 15019)
    assertTrue(AvroUtils.modelContainsIntercept(sc, globalModelPath))

    assertTrue(fs.exists(userModelPath))
    assertModelSane(userModelPath, expectedNumCoefficients = 29, modelId = Some("1436929"))
    assertTrue(AvroUtils.modelContainsIntercept(sc, userModelPath))

    assertTrue(fs.exists(songModelPath))
    assertModelSane(songModelPath, expectedNumCoefficients = 21)
    assertTrue(AvroUtils.modelContainsIntercept(sc, songModelPath))

    assertTrue(fs.exists(artistModelPath))
    assertModelSane(artistModelPath, expectedNumCoefficients = 21)
    assertTrue(AvroUtils.modelContainsIntercept(sc, artistModelPath))

    assertTrue(evaluateModel(new Path(outputDir, "best")) < errorThreshold)
  }

  /**
   * Test that we can calculate feature shard statistics correctly.
   */
  @Test
  def testCalculateFeatureShardStats(): Unit = sparkTest("calculateFeatureShardStats", useKryo = true) {

    val outputDir = new Path(getTmpDir, "output")
    val summarizationDir = new Path(outputDir, "summary")
    val indexMapPath = new Path(getClass.getClassLoader.getResource("GameIntegTest/input/feature-indexes").getPath)
    val fs = outputDir.getFileSystem(sc.hadoopConfiguration)

    runDriver(
      mixedEffectToyRunArgs
        .put(GameTrainingDriver.rootOutputDirectory, outputDir)
        .put(GameTrainingDriver.offHeapIndexMapDirectory, indexMapPath)
        .put(GameTrainingDriver.offHeapIndexMapPartitions, 1)
        .put(GameTrainingDriver.dataSummaryDirectory, summarizationDir))

    mixedEffectFeatureShardConfigs.keys.foreach { featureShardId =>
      assertTrue(fs.exists(new Path(summarizationDir, featureShardId)))
    }
  }

  @DataProvider(name = "badWeightsInputs")
  def badWeightsInputs(): Array[Array[Path]] = {

    val inputDir = getClass.getClassLoader.getResource("DriverIntegTest/input/bad-weights").getPath
    val featuresDir = new Path(inputDir, "feature-lists")

    new File(inputDir)
      .listFiles
      .filter(_.getName.endsWith(".avro"))
      .map(f => Array(featuresDir, new Path(f.getAbsolutePath)))
  }

  /**
   * Test that samples with negative or zero sample weights are filtered out, and throw an exception if there are no
   * samples left after filtering.
   */
  @Test(dataProvider = "badWeightsInputs", expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testBadSampleWeights(featuresDir: Path, inputFile: Path): Unit = sparkTest("testBadSampleWeights") {

    val outputDir = new Path(getTmpDir, "output")

    runDriver(
      fixedEffectToyRunArgs
        .put(GameTrainingDriver.inputDataDirectories, Set(inputFile))
        .put(GameTrainingDriver.rootOutputDirectory, outputDir)
        .put(GameTrainingDriver.featureBagsDirectory, featuresDir)
        .put(GameTrainingDriver.dataValidation, DataValidationType.VALIDATE_FULL))
  }

  /**
   * Perform a very basic sanity check on the model.
   *
   * @param path Path to the model coefficients file
   * @param expectedNumCoefficients Expected number of non-zero coefficients
   * @return True if the model is sane
   */
  def assertModelSane(path: Path, expectedNumCoefficients: Int, modelId: Option[String] = None): Unit = {

    val modelAvro =
      AvroUtils
        .readFromSingleAvro[BayesianLinearModelAvro](sc, path.toString, BayesianLinearModelAvro.getClassSchema.toString)

    val model = modelId match {
      case Some(id) =>
        val m = modelAvro.find { m => m.getModelId.toString == id }
        assertTrue(m.isDefined, s"Model id $id not found.")
        m.get
      case _ => modelAvro.head
    }

    assertEquals(model.getMeans.count(x => x.getValue != 0), expectedNumCoefficients)
  }

  /**
   * Evaluate the model by the specified evaluators with the validation data set.
   *
   * @param modelPath Base path to the GAME model files
   * @return Evaluation results for each specified evaluator
   */
  def evaluateModel(modelPath: Path): Double = {

    val indexMapLoadersOpt = GameTrainingDriver.prepareFeatureMaps()
    val featureSectionMap = GameTrainingDriver
      .getOrDefault(GameTrainingDriver.featureShardConfigurations)
      .mapValues(_.featureBags)
      .map(identity)
    val (testData, indexMapLoaders) = new AvroDataReader(sc).readMerged(
      Seq(testPath.toString),
      indexMapLoadersOpt,
      featureSectionMap,
      numPartitions = 2)
    val partitioner = new LongHashPartitioner(testData.rdd.partitions.length)

    val gameDataSet = GameConverters
      .getGameDataSetFromDataFrame(
        testData,
        featureSectionMap.keySet,
        randomEffectTypes.toSet,
        isResponseRequired = true,
        GameTrainingDriver.getOrDefault(GameTrainingDriver.inputColumnNames))
      .partitionBy(partitioner)

    val validatingLabelsAndOffsetsAndWeights = gameDataSet
      .mapValues(gameData => (gameData.response, gameData.offset, gameData.weight))

    validatingLabelsAndOffsetsAndWeights.count()

    val (gameModel, _) = ModelProcessingUtils.loadGameModelFromHDFS(
      sc,
      modelPath,
      StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL,
      Some(indexMapLoaders))
    val scores = gameModel.score(gameDataSet).scores.mapValues(_.score)

    new RMSEEvaluator(validatingLabelsAndOffsetsAndWeights).evaluate(scores)
  }

  /**
   * Run the GAME training driver with the specified arguments.
   *
   * @param params Arguments for GAME training
   */
  def runDriver(params: ParamMap): Unit = {

    // Reset Driver parameters
    GameTrainingDriver.clear()

    params.toSeq.foreach(paramPair => GameTrainingDriver.set(paramPair.param.asInstanceOf[Param[Any]], paramPair.value))
    GameTrainingDriver.sc = sc
    GameTrainingDriver.logger = new PhotonLogger(
      new Path(GameTrainingDriver.getOrDefault(GameTrainingDriver.rootOutputDirectory), "log"),
      sc)
    GameTrainingDriver.logger.setLogLevel(PhotonLogger.LogLevelDebug)
    GameTrainingDriver.run()
    GameTrainingDriver.logger.close()
  }
}

object GameTrainingDriverIntegTest {

  // This is the Yahoo! Music dataset:
  // photon-ml/photon-client/src/integTest/resources/GameIntegTest/input/train/yahoo-music-train.avro
  private val inputPath = new Path(getClass.getClassLoader.getResource("GameIntegTest/input").getPath)
  private val trainPath = new Path(inputPath, "train")
  private val testPath = new Path(inputPath, "test")
  private val featurePath = new Path(inputPath, "feature-lists")
  private val numExecutors = 1
  private val numIterations = 1

  private val fixedEffectCoordinateId = "global"
  private val fixedEffectFeatureShardId = "shard1"
  private val fixedEffectFeatureShardConfigs = Map(
    (fixedEffectFeatureShardId, FeatureShardConfiguration(Set("features"), hasIntercept = true)))
  private val fixedEffectMinPartitions = numExecutors * 2
  private val fixedEffectDataConfig = FixedEffectDataConfiguration(fixedEffectFeatureShardId, fixedEffectMinPartitions)
  private val fixedEffectOptimizerConfig = OptimizerConfig(
    OptimizerType.TRON,
    maximumIterations = 10,
    tolerance = 1e-5)
  private val fixedEffectOptConfig = FixedEffectOptimizationConfiguration(
    fixedEffectOptimizerConfig,
    L2RegularizationContext)
  private val fixedEffectRegularizationWeights = Set(10D)
  private val fixedEffectCoordinateConfig = FixedEffectCoordinateConfiguration(
    fixedEffectDataConfig,
    fixedEffectOptConfig,
    fixedEffectRegularizationWeights)

  private val randomEffectCoordinateIds = Seq("per-user", "per-song", "per-artist")
  private val randomEffectTypes = Seq("userId", "songId", "artistId")
  private val randomEffectFeatureShardIds = Seq("shard2", "shard3", "shard3")
  private val randomEffectFeatureShardConfigs = Map(
    ("shard2", FeatureShardConfiguration(Set("userFeatures"), hasIntercept = true)),
    ("shard3", FeatureShardConfiguration(Set("songFeatures"), hasIntercept = true)))
  private val randomEffectProjectors = Seq(IndexMapProjection, IndexMapProjection, RandomProjection(2))
  private val randomEffectMinPartitions = numExecutors * 2
  private val randomEffectDataConfigs = randomEffectTypes
    .zip(randomEffectFeatureShardIds)
    .zip(randomEffectProjectors)
    .map { case ((reType, reShardId), reProjector) =>
      RandomEffectDataConfiguration(reType, reShardId, randomEffectMinPartitions, None, None, None, reProjector)
    }
  private val randomEffectOptimizerConfig = OptimizerConfig(
    OptimizerType.TRON,
    maximumIterations = 10,
    tolerance = 1e-5)
  private val randomEffectOptConfig = RandomEffectOptimizationConfiguration(
    randomEffectOptimizerConfig,
    L2RegularizationContext)
  private val randomEffectRegularizationWeights = Set(1D)
  private val randomEffectCoordinateConfigs = randomEffectDataConfigs.map { dataConfig =>
    RandomEffectCoordinateConfiguration(dataConfig, randomEffectOptConfig, randomEffectRegularizationWeights)
  }

  private val mixedEffectFeatureShardConfigs = Map(
    ("shard1", FeatureShardConfiguration(Set("features", "userFeatures", "songFeatures"), hasIntercept = true)),
    ("shard2", FeatureShardConfiguration(Set("features", "userFeatures"), hasIntercept = true)),
    ("shard3", FeatureShardConfiguration(Set("songFeatures"), hasIntercept = true)))

  private val fixedEffectOnlySeriousGameConfig = Map((fixedEffectCoordinateId, fixedEffectCoordinateConfig))
  private val fixedEffectOnlyToyGameConfig = Map(
    (fixedEffectCoordinateId,
      FixedEffectCoordinateConfiguration(
        fixedEffectDataConfig,
        fixedEffectOptConfig.copy(optimizerConfig = fixedEffectOptimizerConfig.copy(maximumIterations = 1)),
        fixedEffectRegularizationWeights)))
  private val randomEffectOnlySeriousGameConfig = Map(randomEffectCoordinateIds.zip(randomEffectCoordinateConfigs): _*)
  private val randomEffectOnlyToyGameConfig = randomEffectOnlySeriousGameConfig.mapValues { reCoordinateConfig =>
    RandomEffectCoordinateConfiguration(
      reCoordinateConfig.dataConfiguration,
      reCoordinateConfig.optimizationConfiguration.copy(
        optimizerConfig = reCoordinateConfig.optimizationConfiguration.optimizerConfig.copy(maximumIterations = 1)),
      reCoordinateConfig.regularizationWeights)
  }
  private val mixedEffectSeriousGameConfig = fixedEffectOnlySeriousGameConfig ++ randomEffectOnlySeriousGameConfig
  private val mixedEffectToyGameConfig = fixedEffectOnlyToyGameConfig ++ randomEffectOnlyToyGameConfig

  /**
   * Default arguments to the GAME training driver.
   *
   * @return Arguments to train a model
   */
  private def defaultArgs: ParamMap =
    ParamMap
      .empty
      .put(GameTrainingDriver.inputDataDirectories, Set(trainPath))
      .put(GameTrainingDriver.validationDataDirectories, Set(testPath))
      .put(GameTrainingDriver.featureBagsDirectory, featurePath)
      .put(GameTrainingDriver.trainingTask, TaskType.LINEAR_REGRESSION)
      .put(GameTrainingDriver.coordinateDescentIterations, numIterations)
      .put(GameTrainingDriver.outputMode, ModelOutputMode.ALL)

  /**
   * Fixed effect arguments with serious optimization. It's useful when we care about the model performance.
   *
   * @return Arguments to train a model
   */
  def fixedEffectSeriousRunArgs: ParamMap =
    defaultArgs
      .put(GameTrainingDriver.featureShardConfigurations, fixedEffectFeatureShardConfigs)
      .put(GameTrainingDriver.coordinateUpdateSequence, Seq(fixedEffectCoordinateId))
      .put(GameTrainingDriver.coordinateConfigurations, fixedEffectOnlySeriousGameConfig)

  /**
   * Fixed effect arguments with "toy" optimization. It's useful when we don't care about the model performance.
   *
   * @return Arguments to train a model
   */
  def fixedEffectToyRunArgs: ParamMap =
    defaultArgs
      .put(GameTrainingDriver.featureShardConfigurations, fixedEffectFeatureShardConfigs)
      .put(GameTrainingDriver.coordinateUpdateSequence, Seq(fixedEffectCoordinateId))
      .put(GameTrainingDriver.coordinateConfigurations, fixedEffectOnlyToyGameConfig)

  /**
   * Random effect arguments with "serious" optimization. It's useful when we care about the model performance.
   *
   * @return Arguments to train a model
   */
  def randomEffectSeriousRunArgs: ParamMap =
    defaultArgs
      .put(GameTrainingDriver.featureShardConfigurations, randomEffectFeatureShardConfigs)
      .put(GameTrainingDriver.coordinateUpdateSequence, randomEffectCoordinateIds)
      .put(GameTrainingDriver.coordinateConfigurations, randomEffectOnlySeriousGameConfig)

  /**
   * Random effect arguments with "toy" optimization. It's useful when we don't care about the model performance.
   *
   * @return Arguments to train a model
   */
  def randomEffectToyRunArgs: ParamMap =
    defaultArgs
      .put(GameTrainingDriver.featureShardConfigurations, randomEffectFeatureShardConfigs)
      .put(GameTrainingDriver.coordinateUpdateSequence, randomEffectCoordinateIds)
      .put(GameTrainingDriver.coordinateConfigurations, randomEffectOnlyToyGameConfig)

  /**
   * Fixed and random effect arguments. It's useful when we care about the model performance.
   *
   * @return Arguments to train a model
   */
  def mixedEffectSeriousRunArgs: ParamMap =
    defaultArgs
      .put(GameTrainingDriver.featureShardConfigurations, mixedEffectFeatureShardConfigs)
      .put(GameTrainingDriver.coordinateUpdateSequence, Seq(fixedEffectCoordinateId) ++ randomEffectCoordinateIds)
      .put(GameTrainingDriver.coordinateConfigurations, mixedEffectSeriousGameConfig)

  /**
   * Fixed and random effect arguments. It's useful when we don't care about the model performance.
   *
   * @return Arguments to train a model
   */
  def mixedEffectToyRunArgs: ParamMap =
    defaultArgs
      .put(GameTrainingDriver.featureShardConfigurations, mixedEffectFeatureShardConfigs)
      .put(GameTrainingDriver.coordinateUpdateSequence, Seq(fixedEffectCoordinateId) ++ randomEffectCoordinateIds)
      .put(GameTrainingDriver.coordinateConfigurations, mixedEffectToyGameConfig)

  /**
   * Build the path to the model coefficients file, given some model properties.
   *
   * @param outputDir Output base directory
   * @param outputMode Output mode (best or all)
   * @param modelType Model type (e.g. "fixed-effect", "random-effect")
   * @param modelName The model name
   * @return Full path to model coefficients file
   */
  private def modelPath(outputDir: Path, outputMode: String, modelType: String, modelName: String): Path =
    new Path(outputDir, s"$outputMode/$modelType/$modelName/coefficients/part-00000.avro")

  /**
   * Build the path to the model coefficients file.
   *
   * @param outputDir Output base directory
   * @param modelType Model type (e.g. "fixed-effect", "random-effect")
   * @param modelName The model name
   * @return Full path to model coefficients file
   */
  def allModelPath(outputDir: Path, modelType: String, modelName: String): Path =
    modelPath(outputDir, "models/0", modelType, modelName)

  /**
   * Build the path to the best model coefficients file.
   *
   * @param outputDir Output base directory
   * @param modelType Model type (e.g. "fixed-effect", "random-effect")
   * @param modelName The model name
   * @return Full path to model coefficients file
   */
  def bestModelPath(outputDir: Path, modelType: String, modelName: String): Path =
    modelPath(outputDir, "best", modelType, modelName)
}
