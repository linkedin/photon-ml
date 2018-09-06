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
import org.apache.spark.storage.StorageLevel
import org.testng.Assert._
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.avro.generated.BayesianLinearModelAvro
import com.linkedin.photon.ml.{DataValidationType, HyperparameterTunerName, HyperparameterTuningMode, TaskType}
import com.linkedin.photon.ml.cli.game.GameDriver
import com.linkedin.photon.ml.constants.MathConst
import com.linkedin.photon.ml.data.{FixedEffectDataConfiguration, GameConverters, RandomEffectDataConfiguration}
import com.linkedin.photon.ml.data.avro._
import com.linkedin.photon.ml.estimators.GameEstimator
import com.linkedin.photon.ml.evaluation.RMSEEvaluator
import com.linkedin.photon.ml.io.{FeatureShardConfiguration, FixedEffectCoordinateConfiguration, ModelOutputMode, RandomEffectCoordinateConfiguration}
import com.linkedin.photon.ml.normalization.NormalizationType
import com.linkedin.photon.ml.optimization._
import com.linkedin.photon.ml.optimization.game.{FixedEffectOptimizationConfiguration, RandomEffectOptimizationConfiguration}
import com.linkedin.photon.ml.projector.{IndexMapProjection, RandomProjection}
import com.linkedin.photon.ml.test.{SparkTestUtils, TestTemplateWithTmpDir}
import com.linkedin.photon.ml.util._

/**
 * Test cases for the GAME training driver.
 *
 * Most of the cases are based on the Yahoo! Music dataset in:
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

    // This is a baseline RMSE capture from an assumed-correct implementation on 01/24/2018
    val errorThreshold = 1.2
    val outputDir = new Path(getTmpDir, "fixedEffects")

    runDriver(
      fixedEffectSeriousRunArgs
        .put(GameTrainingDriver.rootOutputDirectory, outputDir)
        .put(GameTrainingDriver.modelSparsityThreshold, MathConst.EPSILON))

    val allFixedEffectModelPath = outputModelPath(outputDir, AvroConstants.FIXED_EFFECT, fixedEffectCoordinateId)
    val bestFixedEffectModelPath = bestModelPath(outputDir, AvroConstants.FIXED_EFFECT, fixedEffectCoordinateId)
    val fs = outputDir.getFileSystem(sc.hadoopConfiguration)

    assertTrue(fs.exists(allFixedEffectModelPath))
    assertTrue(fs.exists(bestFixedEffectModelPath))

    assertModelSane(allFixedEffectModelPath, expectedNumCoefficients = 14985)
    assertModelSane(bestFixedEffectModelPath, expectedNumCoefficients = 14985)

    assertTrue(evaluateModel(new Path(outputDir, s"${GameTrainingDriver.MODELS_DIR}/0")) < errorThreshold)
    assertTrue(evaluateModel(new Path(outputDir, GameTrainingDriver.BEST_MODEL_DIR)) < errorThreshold)

    assertTrue(AvroUtils.modelContainsIntercept(sc, allFixedEffectModelPath))
    assertTrue(AvroUtils.modelContainsIntercept(sc, bestFixedEffectModelPath))
  }

  /**
   * Test GAME training with a fixed effect model only, and an intercept, and no validation, and only the best model is
   * output.
   *
   * @note Intercepts are optional in [[GameEstimator]], but [[GameDriver]] will setup an intercept by default. This
   *       happens in [[GameDriver.prepareFeatureMapsDefault()]], and there only.
   */
  @Test
  def testFixedEffectsWithAdditionalOpts(): Unit = sparkTest("testFixedEffectsWithIntercept", useKryo = true) {

    // This is a baseline RMSE capture from an assumed-correct implementation on 01/24/2018
    val errorThreshold = 1.2
    val outputDir = new Path(getTmpDir, "fixedEffectsAdditionalOpts")
    val newArgs = fixedEffectSeriousRunArgs
      .copy
      .put(GameTrainingDriver.rootOutputDirectory, outputDir)
      .put(GameTrainingDriver.overrideOutputDirectory, true)
      .put(GameTrainingDriver.outputMode, ModelOutputMode.BEST)
    newArgs.remove(GameTrainingDriver.validationDataDirectories)

    Utils.createHDFSDir(outputDir, sc.hadoopConfiguration)
    runDriver(newArgs)

    val allFixedEffectModelPath = outputModelPath(outputDir, AvroConstants.FIXED_EFFECT, fixedEffectCoordinateId)
    val bestFixedEffectModelPath = bestModelPath(outputDir, AvroConstants.FIXED_EFFECT, fixedEffectCoordinateId)
    val fs = outputDir.getFileSystem(sc.hadoopConfiguration)

    assertFalse(fs.exists(allFixedEffectModelPath))
    assertTrue(fs.exists(bestFixedEffectModelPath))

    assertModelSane(bestFixedEffectModelPath, expectedNumCoefficients = 14983)
    assertTrue(evaluateModel(new Path(outputDir, GameTrainingDriver.BEST_MODEL_DIR)) < errorThreshold)
    assertTrue(AvroUtils.modelContainsIntercept(sc, bestFixedEffectModelPath))
  }

  /**
   * Test GAME training with a fixed effect model only, and no intercept.
   *
   * @note Since intercept terms are ON by default, they need to be explicitly disabled for this test.
   */
  @Test
  def testFixedEffectsWithoutIntercept(): Unit = sparkTest("testFixedEffectsWithoutIntercept", useKryo = true) {

    // This is a baseline RMSE capture from an assumed-correct implementation on 01/24/2018
    val errorThreshold = 1.2
    val outputDir = new Path(getTmpDir, "fixedEffectsNoIntercept")
    val modifiedFeatureShardConfigs = fixedEffectFeatureShardConfigs
      .mapValues(_.copy(hasIntercept = false))
      .map(identity)
    val newArgs = fixedEffectSeriousRunArgs
      .copy
      .put(GameTrainingDriver.rootOutputDirectory, outputDir)
      .put(GameTrainingDriver.featureShardConfigurations, modifiedFeatureShardConfigs)
      .put(GameTrainingDriver.modelSparsityThreshold, MathConst.EPSILON)

    runDriver(newArgs)

    val allFixedEffectModelPath = outputModelPath(outputDir, AvroConstants.FIXED_EFFECT, fixedEffectCoordinateId)
    val bestFixedEffectModelPath = bestModelPath(outputDir, AvroConstants.FIXED_EFFECT, fixedEffectCoordinateId)
    val fs = outputDir.getFileSystem(sc.hadoopConfiguration)

    assertTrue(fs.exists(allFixedEffectModelPath))
    assertTrue(fs.exists(bestFixedEffectModelPath))

    assertModelSane(allFixedEffectModelPath, expectedNumCoefficients = 14984)
    assertModelSane(bestFixedEffectModelPath, expectedNumCoefficients = 14984)

    assertTrue(evaluateModel(new Path(outputDir, s"${GameTrainingDriver.MODELS_DIR}/0")) < errorThreshold)
    assertTrue(evaluateModel(new Path(outputDir, GameTrainingDriver.BEST_MODEL_DIR)) < errorThreshold)

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

    val allFixedEffectModelPath = outputModelPath(outputDir, AvroConstants.FIXED_EFFECT, fixedEffectCoordinateId)
    val bestFixedEffectModelPath = bestModelPath(outputDir, AvroConstants.FIXED_EFFECT, fixedEffectCoordinateId)
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

    val outputDir = new Path(getTmpDir, "fixedEffectsInterceptOnlyBags")
    val modifiedFeatureShardConfigs = fixedEffectFeatureShardConfigs
      .mapValues(_.copy(featureBags = Set()))
      .map(identity)

    runDriver(
      fixedEffectSeriousRunArgs
        .put(GameTrainingDriver.rootOutputDirectory, outputDir)
        .put(GameTrainingDriver.featureShardConfigurations, modifiedFeatureShardConfigs))

    val allFixedEffectModelPath = outputModelPath(outputDir, AvroConstants.FIXED_EFFECT, fixedEffectCoordinateId)
    val bestFixedEffectModelPath = bestModelPath(outputDir, AvroConstants.FIXED_EFFECT, fixedEffectCoordinateId)
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

    // This is a baseline RMSE capture from an assumed-correct implementation on 01/24/2018
    val errorThreshold = 2.34
    val outputDir = new Path(getTmpDir, "randomEffects")

    runDriver(randomEffectSeriousRunArgs.put(GameTrainingDriver.rootOutputDirectory, outputDir))

    val modelPaths = randomEffectCoordinateIds.map(bestModelPath(outputDir, AvroConstants.RANDOM_EFFECT, _))
    val fs = outputDir.getFileSystem(sc.hadoopConfiguration)

    modelPaths.foreach { path =>
      assertTrue(fs.exists(path))
      assertModelSane(path, expectedNumCoefficients = 21)
      assertTrue(AvroUtils.modelContainsIntercept(sc, path))
    }

    assertTrue(evaluateModel(new Path(outputDir, GameTrainingDriver.BEST_MODEL_DIR)) < errorThreshold)
  }

  /**
   * Test GAME training with a random effect models only, and no intercepts.
   */
  @Test
  def testRandomEffectsWithoutAnyIntercept(): Unit = sparkTest("testRandomEffectsWithoutAnyIntercept", useKryo = true) {

    // This is a baseline RMSE capture from an assumed-correct implementation on 01/24/2018
    val errorThreshold = 2.34
    val outputDir = new Path(getTmpDir, "randomEffectsNoIntercept")
    val modifiedFeatureShardConfigs = randomEffectFeatureShardConfigs
      .mapValues(_.copy(hasIntercept = false))
      .map(identity)

    runDriver(
      randomEffectSeriousRunArgs
        .put(GameTrainingDriver.rootOutputDirectory, outputDir)
        .put(GameTrainingDriver.featureShardConfigurations, modifiedFeatureShardConfigs))

    val modelPaths = randomEffectCoordinateIds.map(bestModelPath(outputDir, AvroConstants.RANDOM_EFFECT, _))
    val fs = outputDir.getFileSystem(sc.hadoopConfiguration)

    modelPaths.foreach { path =>
      assertTrue(fs.exists(path))
      assertModelSane(path, expectedNumCoefficients = 20)
      assertFalse(AvroUtils.modelContainsIntercept(sc, path))
    }

    assertTrue(evaluateModel(new Path(outputDir, GameTrainingDriver.BEST_MODEL_DIR)) < errorThreshold)
  }

  /**
   * Test GAME training with a random effect models with normalization and index-map projection
   */
  @Test
  def testRandomEffectWithNormalization(): Unit = sparkTest("testRandomEffectsWithoutAnyIntercept", useKryo = true) {

    // This is a baseline RMSE capture from an assumed-correct implementation on 01/24/2018
    val errorThreshold = 2.34
    val outputDir = new Path(getTmpDir, "randomEffectsNormalization")
    runDriver(
      randomEffectSeriousRunArgs
        .put(GameTrainingDriver.rootOutputDirectory, outputDir)
        .put(GameTrainingDriver.normalization, NormalizationType.SCALE_WITH_MAX_MAGNITUDE))

    val modelPaths = randomEffectCoordinateIds.map(bestModelPath(outputDir, AvroConstants.RANDOM_EFFECT, _))
    val fs = outputDir.getFileSystem(sc.hadoopConfiguration)

    modelPaths.foreach { path =>
      assertTrue(fs.exists(path))
      assertModelSane(path, expectedNumCoefficients = 21)
      assertTrue(AvroUtils.modelContainsIntercept(sc, path))
    }

    assertTrue(evaluateModel(new Path(outputDir, GameTrainingDriver.BEST_MODEL_DIR)) < errorThreshold)
  }

  /**
   * Test GAME training with a both fixed and random effect models.
   */
  @Test
  def testFixedAndRandomEffects(): Unit = sparkTest("fixedAndRandomEffects", useKryo = true) {

    // This is a baseline RMSE capture from an assumed-correct implementation on 01/24/2018
    val errorThreshold = 0.95
    val outputDir = new Path(getTmpDir, "fixedAndRandomEffects")

    runDriver(mixedEffectSeriousRunArgs.put(GameTrainingDriver.rootOutputDirectory, outputDir))

    val globalModelPath = bestModelPath(outputDir, AvroConstants.FIXED_EFFECT, "global")
    val userModelPath = bestModelPath(outputDir, AvroConstants.RANDOM_EFFECT, "per-user")
    val songModelPath = bestModelPath(outputDir, AvroConstants.RANDOM_EFFECT, "per-song")
    val artistModelPath = bestModelPath(outputDir, AvroConstants.RANDOM_EFFECT, "per-artist")
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

    assertTrue(evaluateModel(new Path(outputDir, GameTrainingDriver.BEST_MODEL_DIR)) < errorThreshold)
  }

  /**
   * Test GAME training with a fixed effect model only and hyperparameter tuning. Note that the best model may not be
   * one of the tuned models. (This test is commented out since hyperparameter tuning is temporarily disabled in
   * photon-ml. Hyperparameter tuning is still available in LinkedIn internal library li-photon-ml.)
   */
//  @Test
//  def c(): Unit = sparkTest("testHyperParameterTuning", useKryo = true) {
//
//    val hyperParameterTuningIter = 1
//    val outputDir = new Path(getTmpDir, "hyperParameterTuning")
//    val newArgs = mixedEffectSeriousRunArgs
//      .copy
//      .put(GameTrainingDriver.rootOutputDirectory, outputDir)
//      .put(GameTrainingDriver.outputMode, ModelOutputMode.TUNED)
//      .put(GameTrainingDriver.hyperParameterTunerName, HyperparameterTunerName.DUMMY)
//      .put(GameTrainingDriver.hyperParameterTuning, HyperparameterTuningMode.RANDOM)
//      .put(GameTrainingDriver.hyperParameterTuningIter, hyperParameterTuningIter)
//
//    runDriver(newArgs)
//
//    val allModelsPath = new Path(outputDir, s"${GameTrainingDriver.MODELS_DIR}")
//    val bestModelPath = new Path(outputDir, s"${GameTrainingDriver.BEST_MODEL_DIR}")
//    val fs = outputDir.getFileSystem(sc.hadoopConfiguration)
//
//    assertTrue(fs.exists(allModelsPath))
//    assertTrue(fs.exists(bestModelPath))
//
//    val allFixedEffectModelsPathContents = fs.listStatus(allModelsPath)
//    assertEquals(allFixedEffectModelsPathContents.length, fixedEffectRegularizationWeights.size)
//    allFixedEffectModelsPathContents.forall(_.isDirectory)
//
//    val bestRMSE = evaluateModel(bestModelPath)
//
//    (0 until hyperParameterTuningIter).foreach { i =>
//      val modelPath = new Path(allModelsPath, s"$i")
//      val fixedEffectModelPath = outputModelPath(outputDir, AvroConstants.FIXED_EFFECT, fixedEffectCoordinateId, i)
//      assertTrue(fs.exists(fixedEffectModelPath))
//
//      randomEffectCoordinateIds.foreach { randomEffectCoordinateId =>
//        val randomEffectModelPath = outputModelPath(outputDir, AvroConstants.RANDOM_EFFECT, randomEffectCoordinateId, i)
//        assertTrue(fs.exists(randomEffectModelPath))
//      }
//
//      assertTrue(evaluateModel(modelPath) >= bestRMSE)
//    }
//  }

  /**
   * Test GAME partial retraining using a pre-trained fixed effect model.
   */
  @Test
  def testPartialRetrainWithFixedBase(): Unit = sparkTest("testPartialRetrainWithFixedBase", useKryo = true) {

    val outputDir = new Path(getTmpDir, "testPartialRetrainWithFixedBase")

    runDriver(partialRetrainWithFixedBaseArgs.put(GameTrainingDriver.rootOutputDirectory, outputDir))

    compareModelEvaluation(new Path(outputDir, "best"), trainedMixedModelPath, TOLERANCE)
  }

  /**
   * Test GAME partial retraining using a pre-trained random effects model.
   */
  @Test(enabled = false)
  def testPartialRetrainWithRandomBase(): Unit = sparkTest("testPartialRetrainWithRandomBase", useKryo = true) {

    // TODO: Currently this test fails because in a full re-training scenario, the scores for a coordinate start out
    // TODO: assuming all-zero models for each coordinate, and updated scores are added as the coordinates are trained.
    // TODO: However, with the current partial re-training code, the scores for the pre-trained coordinates are
    // TODO: immediately calculated and used, thus changing the offsets that coordinates are trained with. Need to
    // TODO: evaluate whether this is an issue or not. Can this degrade the training performance? The assumption of
    // TODO: coordinate descent is that doing this actually improves performance, but order does matter.
    val outputDir = new Path(getTmpDir, "testPartialRetrainWithRandomBase")

    runDriver(partialRetrainWithRandomBaseArgs.put(GameTrainingDriver.rootOutputDirectory, outputDir))

    compareModelEvaluation(new Path(outputDir, "best"), trainedMixedModelPath, TOLERANCE)
  }

  /**
   * Test GAME warm-start with initial model
   */
  @Test
  def testWarmStartWithInitialModel(): Unit = sparkTest("testWarmStartWithInitialModel", useKryo = true) {
    // TODO This is really not much more than a quick sanity check. At some point we need to split the data in 2 sets,
    // and check that incremental training works as expected.
    val outputDir = new Path(getTmpDir, "testInitialModel")

    val args = defaultArgs
      .put(GameTrainingDriver.featureShardConfigurations, mixedEffectFeatureShardConfigs)
      .put(GameTrainingDriver.coordinateUpdateSequence, Seq(fixedEffectCoordinateId) ++ randomEffectCoordinateIds)
      .put(GameTrainingDriver.coordinateConfigurations, mixedEffectSeriousGameConfig)
      .put(GameTrainingDriver.modelInputDirectory, trainedMixedModelPath)

    runDriver(args.put(GameTrainingDriver.rootOutputDirectory, outputDir))

    compareModelEvaluation(new Path(outputDir, "best"), trainedMixedModelPath, 0.005)
  }

  /**
   * Test GAME training with a custom model sparsity threshold.
   */
  @Test
  def testModelSparsityThreshold(): Unit = sparkTest("testModelSparsityThreshold", useKryo = true) {

    val outputDir = new Path(getTmpDir, "testModelSparsityThreshold")

    runDriver(
      mixedEffectSeriousRunArgs
        .put(GameTrainingDriver.rootOutputDirectory, outputDir)
        .put(GameTrainingDriver.modelSparsityThreshold, 100.0))

    val globalModelPath = bestModelPath(outputDir, AvroConstants.FIXED_EFFECT, "global")
    val userModelPath = bestModelPath(outputDir, AvroConstants.RANDOM_EFFECT, "per-user")
    val songModelPath = bestModelPath(outputDir, AvroConstants.RANDOM_EFFECT, "per-song")
    val artistModelPath = bestModelPath(outputDir, AvroConstants.RANDOM_EFFECT, "per-artist")
    val fs = outputDir.getFileSystem(sc.hadoopConfiguration)

    assertTrue(fs.exists(globalModelPath))
    assertModelSane(globalModelPath, expectedNumCoefficients = 0)

    assertTrue(fs.exists(userModelPath))
    assertModelSane(userModelPath, expectedNumCoefficients = 0, modelId = Some("1436929"))

    assertTrue(fs.exists(songModelPath))
    assertModelSane(songModelPath, expectedNumCoefficients = 0)

    assertTrue(fs.exists(artistModelPath))
    assertModelSane(artistModelPath, expectedNumCoefficients = 0)
  }

  /**
   * Test GAME training, loading an off-heap index map.
   */
  @Test
  def testOffHeapIndexMap(): Unit = sparkTest("testOffHeapIndexMap", useKryo = true) {

    val outputDir = new Path(getTmpDir, "testOffHeapIndexMap")
    val indexMapPath = new Path(getClass.getClassLoader.getResource("GameIntegTest/input/feature-indexes").getPath)
    val params = mixedEffectToyRunArgs
      .put(GameTrainingDriver.rootOutputDirectory, outputDir)
      .put(GameTrainingDriver.offHeapIndexMapDirectory, indexMapPath)
      .put(GameTrainingDriver.offHeapIndexMapPartitions, 1)
    params.remove(GameTrainingDriver.featureBagsDirectory)

    runDriver(params)

    val globalModelPath = bestModelPath(outputDir, AvroConstants.FIXED_EFFECT, "global")
    val userModelPath = bestModelPath(outputDir, AvroConstants.RANDOM_EFFECT, "per-user")
    val songModelPath = bestModelPath(outputDir, AvroConstants.RANDOM_EFFECT, "per-song")
    val artistModelPath = bestModelPath(outputDir, AvroConstants.RANDOM_EFFECT, "per-artist")
    val fs = outputDir.getFileSystem(sc.hadoopConfiguration)

    assertTrue(fs.exists(globalModelPath))
    assertTrue(fs.exists(userModelPath))
    assertTrue(fs.exists(songModelPath))
    assertTrue(fs.exists(artistModelPath))
  }

  /**
   * Test that we can calculate feature shard statistics correctly.
   */
  @Test
  def testCalculateFeatureShardStats(): Unit = sparkTest("calculateFeatureShardStats", useKryo = true) {

    val outputDir = new Path(getTmpDir, "output")
    val summarizationDir = new Path(outputDir, "summary")
    val fs = outputDir.getFileSystem(sc.hadoopConfiguration)

    runDriver(
      mixedEffectToyRunArgs
        .put(GameTrainingDriver.rootOutputDirectory, outputDir)
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

    val modelAvro = AvroUtils.readFromSingleAvro[BayesianLinearModelAvro](
      sc,
      path.toString,
      BayesianLinearModelAvro.getClassSchema.toString)

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
   * Compare the RMSE evaluation results of two models.
   *
   * @param modelPath1 Base path to the GAME model files of the first model
   * @param modelPath2 Base path to the GAME model files of the second model
   * @param tolerance The tolerance within the RMSE of the two models should match
   */
  def compareModelEvaluation(modelPath1: Path, modelPath2: Path, tolerance: Double): Unit = {

    val indexMapLoadersOpt = GameTrainingDriver.prepareFeatureMaps()
    val featureShardConfigs = GameTrainingDriver.getOrDefault(GameTrainingDriver.featureShardConfigurations)
    val (testData, indexMapLoaders) = new AvroDataReader().readMerged(
      Seq(testPath.toString),
      indexMapLoadersOpt,
      featureShardConfigs,
      numPartitions = 2)
    val partitioner = new LongHashPartitioner(testData.rdd.partitions.length)

    val gameDataSet = GameConverters
      .getGameDataSetFromDataFrame(
        testData,
        featureShardConfigs.keySet,
        randomEffectTypes.toSet,
        isResponseRequired = true,
        GameTrainingDriver.getOrDefault(GameTrainingDriver.inputColumnNames))
      .partitionBy(partitioner)

    val validatingLabelsAndOffsetsAndWeights = gameDataSet
      .mapValues(gameData => (gameData.response, gameData.offset, gameData.weight))

    validatingLabelsAndOffsetsAndWeights.count()

    val gameModel1 = ModelProcessingUtils.loadGameModelFromHDFS(
      sc,
      modelPath1,
      StorageLevel.DISK_ONLY,
      indexMapLoaders)
    val gameModel2 = ModelProcessingUtils.loadGameModelFromHDFS(
      sc,
      modelPath2,
      StorageLevel.DISK_ONLY,
      indexMapLoaders)

    val scores1 = gameModel1.score(gameDataSet).scores.mapValues(_.score)
    val scores2 = gameModel2.score(gameDataSet).scores.mapValues(_.score)

    val rmseEval = new RMSEEvaluator(validatingLabelsAndOffsetsAndWeights)
    val rmse1 = rmseEval.evaluate(scores1)
    val rmse2 = rmseEval.evaluate(scores2)

    assertEquals(rmse1, rmse2, tolerance)
  }

  /**
   * Evaluate the model by the specified evaluators with the validation dataset.
   *
   * @param modelPath Base path to the GAME model files
   * @return Evaluation results for each specified evaluator
   */
  def evaluateModel(modelPath: Path): Double = {

    val indexMapLoadersOpt = GameTrainingDriver.prepareFeatureMaps()
    val featureShardConfigs = GameTrainingDriver.getOrDefault(GameTrainingDriver.featureShardConfigurations)
    val (testData, indexMapLoaders) = new AvroDataReader().readMerged(
      Seq(testPath.toString),
      indexMapLoadersOpt,
      featureShardConfigs,
      numPartitions = 2)
    val partitioner = new LongHashPartitioner(testData.rdd.partitions.length)

    val gameDataSet = GameConverters
      .getGameDataSetFromDataFrame(
        testData,
        featureShardConfigs.keySet,
        randomEffectTypes.toSet,
        isResponseRequired = true,
        GameTrainingDriver.getOrDefault(GameTrainingDriver.inputColumnNames))
      .partitionBy(partitioner)

    val validatingLabelsAndOffsetsAndWeights = gameDataSet
      .mapValues(gameData => (gameData.response, gameData.offset, gameData.weight))

    validatingLabelsAndOffsetsAndWeights.count()

    val gameModel = ModelProcessingUtils.loadGameModelFromHDFS(
      sc,
      modelPath,
      StorageLevel.DISK_ONLY,
      indexMapLoaders)

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

  private val TOLERANCE = 1E-6

  // This is the Yahoo! Music dataset:
  // photon-ml/photon-client/src/integTest/resources/GameIntegTest/input/train/yahoo-music-train.avro
  private val basePath = new Path(getClass.getClassLoader.getResource("GameIntegTest").getPath)
  private val inputPath = new Path(basePath, "input")
  private val trainPath = new Path(inputPath, "train")
  private val testPath = new Path(inputPath, "test")
  private val featurePath = new Path(inputPath, "feature-lists")
  private val trainedModelsPath = new Path(basePath, "retrainModels")
  private val trainedFixedOnlyModelPath = new Path(trainedModelsPath, "fixedEffectsOnly")
  private val trainedRandomOnlyModelPath = new Path(trainedModelsPath, "randomEffectsOnly")
  private val trainedMixedModelPath = new Path(trainedModelsPath, "mixedEffects")
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
    tolerance = 1e-5,
    constraintMap = None)
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
      RandomEffectDataConfiguration(reType, reShardId, randomEffectMinPartitions, projectorType = reProjector)
    }
  private val randomEffectOptimizerConfig = OptimizerConfig(
    OptimizerType.TRON,
    maximumIterations = 10,
    tolerance = 1e-5,
    constraintMap = None)
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
   * Fixed and random effect arguments, but retraining using an existing fixed effect model.
   *
   * @return Arguments to train a model
   */
  def partialRetrainWithFixedBaseArgs: ParamMap =
    defaultArgs
      .put(GameTrainingDriver.featureShardConfigurations, mixedEffectFeatureShardConfigs)
      .put(GameTrainingDriver.coordinateUpdateSequence, Seq(fixedEffectCoordinateId) ++ randomEffectCoordinateIds)
      .put(GameTrainingDriver.coordinateConfigurations, randomEffectOnlySeriousGameConfig)
      .put(GameTrainingDriver.modelInputDirectory, trainedFixedOnlyModelPath)
      .put(GameTrainingDriver.partialRetrainLockedCoordinates, Set(fixedEffectCoordinateId))

  /**
   * Fixed and random effect arguments, but retraining using an existing random effects model.
   *
   * @return Arguments to train a model
   */
  def partialRetrainWithRandomBaseArgs: ParamMap =
    defaultArgs
      .put(GameTrainingDriver.featureShardConfigurations, mixedEffectFeatureShardConfigs)
      .put(GameTrainingDriver.coordinateUpdateSequence, Seq(fixedEffectCoordinateId) ++ randomEffectCoordinateIds)
      .put(GameTrainingDriver.coordinateConfigurations, fixedEffectOnlySeriousGameConfig)
      .put(GameTrainingDriver.modelInputDirectory, trainedRandomOnlyModelPath)
      .put(GameTrainingDriver.partialRetrainLockedCoordinates, randomEffectCoordinateIds.toSet)

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
  def outputModelPath(outputDir: Path, modelType: String, modelName: String, modelPos: Int = 0): Path =
    modelPath(outputDir, s"${GameTrainingDriver.MODELS_DIR}/$modelPos", modelType, modelName)

  /**
   * Build the path to the best model coefficients file.
   *
   * @param outputDir Output base directory
   * @param modelType Model type (e.g. "fixed-effect", "random-effect")
   * @param modelName The model name
   * @return Full path to model coefficients file
   */
  def bestModelPath(outputDir: Path, modelType: String, modelName: String): Path =
    modelPath(outputDir, GameTrainingDriver.BEST_MODEL_DIR, modelType, modelName)
}
