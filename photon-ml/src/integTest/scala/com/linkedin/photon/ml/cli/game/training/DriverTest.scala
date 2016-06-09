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
package com.linkedin.photon.ml.cli.game.training

import collection.JavaConversions._
import com.linkedin.photon.ml.SparkContextConfiguration
import com.linkedin.photon.ml.avro.AvroIOUtils
import com.linkedin.photon.ml.avro.data.NameAndTerm
import com.linkedin.photon.ml.avro.generated.BayesianLinearModelAvro
import com.linkedin.photon.ml.avro.model.ModelProcessingUtils
import com.linkedin.photon.ml.data.{FixedEffectDataSet, RandomEffectDataSet}
import com.linkedin.photon.ml.io.ModelOutputMode
import com.linkedin.photon.ml.supervised.TaskType
import com.linkedin.photon.ml.supervised.regression.LinearRegressionModel
import com.linkedin.photon.ml.test.{CommonTestUtils, SparkTestUtils, TestTemplateWithTmpDir}
import com.linkedin.photon.ml.util.{Utils, PhotonLogger}
import org.apache.spark.SparkConf
import org.testng.annotations.Test
import org.testng.Assert._

import java.nio.file.{Files, FileSystems, Path}

class DriverTest extends SparkTestUtils with TestTemplateWithTmpDir {
  import DriverTest._
  import CommonTestUtils._

  @Test
  def testFixedEffectsWithIntercept() = sparkTest("testFixedEffectsWithIntercept", useKryo = true) {
    val outputDir = s"$getTmpDir/fixedEffects"
    // This is a baseline RMSE capture from an assumed-correct implementation on 4/14/2016
    val errorThreshold = 1.7
    val driver = runDriver(argArray(fixedEffectSeriousRunArgs ++ Map("output-dir" -> outputDir)))
    val allFixedEffectModelPath = allModelPath(outputDir, "fixed-effect", "global")
    val bestFixedEffectModelPath = bestModelPath(outputDir, "fixed-effect", "global")

    println(allFixedEffectModelPath)
    assertTrue(Files.exists(allFixedEffectModelPath))
    assertTrue(Files.exists(bestFixedEffectModelPath))
    assertTrue(modelSane(allFixedEffectModelPath, expectedNumCoefficients = 14983))
    assertTrue(modelSane(bestFixedEffectModelPath, expectedNumCoefficients = 14983))
    assertTrue(evaluateModel(driver, fs.getPath(outputDir, "all/0")) < errorThreshold)
    assertTrue(evaluateModel(driver, fs.getPath(outputDir, "best")) < errorThreshold)
    assertTrue(modelContainsIntercept(allFixedEffectModelPath))
    assertTrue(modelContainsIntercept(bestFixedEffectModelPath))
  }

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def failedTestRunWithOutputDirExists(): Unit = sparkTest("failedTestRunWithOutputDirExists") {
    val outputDir = getTmpDir + "/failedTestRunWithOutputDirExists"
    Utils.createHDFSDir(outputDir, sc.hadoopConfiguration)
    runDriver(argArray(fixedEffectToyRunArgs ++ Map("output-dir" -> outputDir)))
  }

  @Test
  def successfulTestRunWithOutputDirExists(): Unit = sparkTest("successfulTestRunWithOutputDirExists") {
    val outputDir = getTmpDir + "/successfulTestRunWithOutputDirExists"
    Utils.createHDFSDir(outputDir, sc.hadoopConfiguration)
    runDriver(argArray(fixedEffectToyRunArgs ++
        Map("output-dir" -> outputDir, "delete-output-dir-if-exists" -> "true")))
  }

  @Test
  def testFixedEffectsWithoutIntercept() = sparkTest("testFixedEffectsWithoutIntercept", useKryo = true) {
    val outputDir = s"$getTmpDir/fixedEffects"
    runDriver(argArray(fixedEffectToyRunArgs ++ Map("feature-shard-id-to-intercept-map" -> "shard1:false")
        ++ Map("output-dir" -> outputDir)))
    val allFixedEffectModelPath = allModelPath(outputDir, "fixed-effect", "global")
    val bestFixedEffectModelPath = bestModelPath(outputDir, "fixed-effect", "global")

    assertTrue(Files.exists(allFixedEffectModelPath))
    assertTrue(Files.exists(bestFixedEffectModelPath))
    assertTrue(modelSane(allFixedEffectModelPath, expectedNumCoefficients = 11597))
    assertTrue(modelSane(bestFixedEffectModelPath, expectedNumCoefficients = 11597))
    assertFalse(modelContainsIntercept(allFixedEffectModelPath))
    assertFalse(modelContainsIntercept(bestFixedEffectModelPath))
  }

  @Test
  def testSaveBestOnly() = sparkTest("saveBestOnly", useKryo = true) {
    val outputDir = s"$getTmpDir/fixedEffects"
    runDriver(argArray(fixedEffectToyRunArgs ++ Map(
      "output-dir" -> outputDir,
      "model-output-mode" -> ModelOutputMode.BEST.toString)))
    val allFixedEffectModelPath = allModelPath(outputDir, "fixed-effect", "global")
    val bestFixedEffectModelPath = bestModelPath(outputDir, "fixed-effect", "global")

    assertFalse(Files.exists(allFixedEffectModelPath))
    assertTrue(Files.exists(bestFixedEffectModelPath))
  }

  @Test
  def testSaveNone() = sparkTest("saveNone", useKryo = true) {
    val outputDir = s"$getTmpDir/fixedEffects"
    runDriver(argArray(fixedEffectToyRunArgs ++ Map(
      "output-dir" -> outputDir,
      "model-output-mode" -> ModelOutputMode.NONE.toString)))
    val allFixedEffectModelPath = allModelPath(outputDir, "fixed-effect", "global")
    val bestFixedEffectModelPath = bestModelPath(outputDir, "fixed-effect", "global")

    assertFalse(Files.exists(allFixedEffectModelPath))
    assertFalse(Files.exists(bestFixedEffectModelPath))
  }

  @Test
  def testRandomEffectsWithIntercept() = sparkTest("testRandomEffectsWithIntercept", useKryo = true) {
    val outputDir = s"$getTmpDir/randomEffects"
    // This is a baseline RMSE capture from an assumed-correct implementation on 4/14/2016
    val errorThreshold = 2.2
    val driver = runDriver(argArray(randomEffectSeriousRunArgs ++ Map("output-dir" -> outputDir)))
    val userModelPath = bestModelPath(outputDir, "random-effect", "per-user")
    val songModelPath = bestModelPath(outputDir, "random-effect", "per-song")
    val artistModelPath = bestModelPath(outputDir, "random-effect", "per-artist")

    assertTrue(Files.exists(userModelPath))
    assertTrue(modelSane(userModelPath, expectedNumCoefficients = 21))
    assertTrue(modelContainsIntercept(userModelPath))

    assertTrue(Files.exists(songModelPath))
    assertTrue(modelSane(songModelPath, expectedNumCoefficients = 21))
    assertTrue(modelContainsIntercept(songModelPath))

    assertTrue(Files.exists(artistModelPath))
    assertTrue(modelSane(artistModelPath, expectedNumCoefficients = 21))
    assertTrue(modelContainsIntercept(artistModelPath))

    assertTrue(evaluateModel(driver, fs.getPath(outputDir, "best")) < errorThreshold)
  }

  @Test
  def testRandomEffectsWithoutAnyIntercept() = sparkTest("testRandomEffectsWithoutAnyIntercept", useKryo = true) {
    val outputDir = s"$getTmpDir/randomEffects"
    runDriver(argArray(randomEffectToyRunArgs ++
        Map("feature-shard-id-to-intercept-map" -> "shard2:false|shard3:false") ++ Map("output-dir" -> outputDir)))
    val userModelPath = bestModelPath(outputDir, "random-effect", "per-user")
    val songModelPath = bestModelPath(outputDir, "random-effect", "per-song")
    val artistModelPath = bestModelPath(outputDir, "random-effect", "per-artist")

    assertTrue(Files.exists(userModelPath))
    assertTrue(modelSane(userModelPath, expectedNumCoefficients = 20))
    assertFalse(modelContainsIntercept(userModelPath))

    assertTrue(Files.exists(songModelPath))
    assertTrue(modelSane(songModelPath, expectedNumCoefficients = 20))
    assertFalse(modelContainsIntercept(songModelPath))

    assertTrue(Files.exists(artistModelPath))
    assertTrue(modelSane(artistModelPath, expectedNumCoefficients = 20))
    assertFalse(modelContainsIntercept(artistModelPath))
  }

  @Test
  def testRandomEffectsWithPartialIntercept() = sparkTest("testRandomEffectsWithPartialIntercept", useKryo = true) {
    val outputDir = s"$getTmpDir/randomEffects"
    runDriver(argArray(randomEffectToyRunArgs ++
        Map("feature-shard-id-to-intercept-map" -> "shard2:false|shard3:true") ++ Map("output-dir" -> outputDir)))
    val userModelPath = bestModelPath(outputDir, "random-effect", "per-user")
    val songModelPath = bestModelPath(outputDir, "random-effect", "per-song")
    val artistModelPath = bestModelPath(outputDir, "random-effect", "per-artist")

    assertTrue(Files.exists(userModelPath))
    assertTrue(modelSane(userModelPath, expectedNumCoefficients = 20))
    assertFalse(modelContainsIntercept(userModelPath))

    assertTrue(Files.exists(songModelPath))
    assertTrue(modelSane(songModelPath, expectedNumCoefficients = 21))
    assertTrue(modelContainsIntercept(songModelPath))

    assertTrue(Files.exists(artistModelPath))
    assertTrue(modelSane(artistModelPath, expectedNumCoefficients = 21))
    assertTrue(modelContainsIntercept(artistModelPath))
  }

  @Test
  def testFixedAndRandomEffects() = sparkTest("fixedAndRandomEffects", useKryo = true) {
    val outputDir = s"$getTmpDir/fixedAndRandomEffects"

    // This is a baseline RMSE capture from an assumed-correct implementation on 4/14/2016
    val errorThreshold = 2.2

    val driver = runDriver(argArray(fixedAndRandomEffectSeriousRunArgs ++ Map("output-dir" -> outputDir)))

    val fixedEffectModelPath = bestModelPath(outputDir, "fixed-effect", "global")
    val userModelPath = bestModelPath(outputDir, "random-effect", "per-user")
    val songModelPath = bestModelPath(outputDir, "random-effect", "per-song")
    val artistModelPath = bestModelPath(outputDir, "random-effect", "per-artist")

    assertTrue(Files.exists(fixedEffectModelPath))
    assertTrue(modelSane(fixedEffectModelPath, expectedNumCoefficients = 15017))
    assertTrue(modelContainsIntercept(fixedEffectModelPath))

    assertTrue(Files.exists(userModelPath))
    assertTrue(modelSane(userModelPath, expectedNumCoefficients = 29))
    assertTrue(modelContainsIntercept(userModelPath))

    assertTrue(Files.exists(songModelPath))
    assertTrue(modelSane(songModelPath, expectedNumCoefficients = 21))
    assertTrue(modelContainsIntercept(songModelPath))

    assertTrue(Files.exists(artistModelPath))
    assertTrue(modelSane(artistModelPath, expectedNumCoefficients = 21))
    assertTrue(modelContainsIntercept(artistModelPath))

    assertTrue(evaluateModel(driver, fs.getPath(outputDir, "best")) < errorThreshold)
  }

  @Test
  def testPrepareFixedEffectTrainingDataSet() = sparkTest("prepareFixedEffectTrainingDataSet", useKryo = true) {
    val outputDir = s"$getTmpDir/prepareFixedEffectTrainingDataSet"

    val args = argArray(fixedEffectToyRunArgs ++ Map(
      "output-dir" -> outputDir))

    val driver = new Driver(
      Params.parseFromCommandLine(args), sc, new PhotonLogger(s"$outputDir/log", sc))

    val featureShardIdToFeatureMapMap = driver.prepareFeatureMaps()
    val gameDataSet = driver.prepareGameDataSet(featureShardIdToFeatureMapMap)
    val trainingDataSet = driver.prepareTrainingDataSet(gameDataSet)

    assertEquals(trainingDataSet.size, 1)

    trainingDataSet("global") match {
      case ds: FixedEffectDataSet =>
        assertEquals(ds.labeledPoints.count(), 34810)
        assertEquals(ds.numFeatures, 30045)

      case _ => fail("Wrong dataset type.")
    }
  }

  @Test
  def testPrepareFixedAndRandomEffectTrainingDataSet() =
    sparkTest("prepareFixedAndRandomEffectTrainingDataSet", useKryo = true) {
      val outputDir = s"$getTmpDir/prepareFixedEffectTrainingDataSet"

      val args = argArray(fixedAndRandomEffectToyRunArgs ++ Map(
        "output-dir" -> outputDir))

      val driver = new Driver(
        Params.parseFromCommandLine(args), sc, new PhotonLogger(s"$outputDir/log", sc))

      val featureShardIdToFeatureMapMap = driver.prepareFeatureMaps()
      val gameDataSet = driver.prepareGameDataSet(featureShardIdToFeatureMapMap)
      val trainingDataSet = driver.prepareTrainingDataSet(gameDataSet)

      assertEquals(trainingDataSet.size, 4)

      // fixed effect data
      trainingDataSet("global") match {
        case ds: FixedEffectDataSet =>
          assertEquals(ds.labeledPoints.count(), 34810)
          assertEquals(ds.numFeatures, 30085)

        case _ => fail("Wrong dataset type.")
      }

      // per-user data
      trainingDataSet("per-user") match {
        case ds: RandomEffectDataSet =>
          assertEquals(ds.activeData.count(), 33110)

          val featureStats = ds.activeData.values.map(_.numActiveFeatures).stats()
          assertEquals(featureStats.count, 33110)
          assertEquals(featureStats.mean, 24.12999, tol)
          assertEquals(featureStats.stdev, 0.611194, tol)
          assertEquals(featureStats.max, 40.0, tol)
          assertEquals(featureStats.min, 24.0, tol)

        case _ => fail("Wrong dataset type.")
      }

      // per-song data
      trainingDataSet("per-song") match {
        case ds: RandomEffectDataSet =>
          assertEquals(ds.activeData.count(), 23167)

          val featureStats = ds.activeData.values.map(_.numActiveFeatures).stats()
          assertEquals(featureStats.count, 23167)
          assertEquals(featureStats.mean, 21.0, tol)
          assertEquals(featureStats.stdev, 0.0, tol)
          assertEquals(featureStats.max, 21.0, tol)
          assertEquals(featureStats.min, 21.0, tol)

        case _ => fail("Wrong dataset type.")
      }

      // per-artist data
      trainingDataSet("per-artist") match {
        case ds: RandomEffectDataSet =>
          assertEquals(ds.activeData.count(), 4471)

          val featureStats = ds.activeData.values.map(_.numActiveFeatures).stats()
          assertEquals(featureStats.count, 4471)
          assertEquals(featureStats.mean, 3.0, tol)
          assertEquals(featureStats.stdev, 0.0, tol)
          assertEquals(featureStats.max, 3.0, tol)
          assertEquals(featureStats.min, 3.0, tol)

        case _ => fail("Wrong dataset type.")
      }
    }

  @Test
  def testMultipleOptimizerConfigs() = sparkTest("multipleOptimizerConfigs", useKryo = true) {
    val outputDir = s"$getTmpDir/multipleOptimizerConfigs"

    // This is a baseline RMSE capture from an assumed-correct implementation on 4/14/2016
    val errorThreshold = 1.7

    val driver = runDriver(argArray(fixedEffectSeriousRunArgs ++ Map(
      "output-dir" -> outputDir,
      "fixed-effect-optimization-configurations" ->
        ("global:10,1e-5,10,1,tron,l2;" +
          "global:10,1e-5,10,1,lbfgs,l2"))))

    val fixedEffectModelPath = bestModelPath(outputDir, "fixed-effect", "global")

    assertTrue(Files.exists(fixedEffectModelPath))
    assertTrue(modelSane(fixedEffectModelPath, expectedNumCoefficients = 14983))
    assertTrue(evaluateModel(driver, fs.getPath(outputDir, "best")) < errorThreshold)
  }

  /**
    * Overridden spark test provider that allows for specifying whether to use kryo serialization
    *
    * @param name the test job name
    * @param body the execution closure
    */
  def sparkTest(name: String, useKryo: Boolean)(body: => Unit) {
    SparkTestUtils.SPARK_LOCAL_CONFIG.synchronized {
      sc = SparkContextConfiguration.asYarnClient(
        new SparkConf().setMaster(SparkTestUtils.SPARK_LOCAL_CONFIG), name, useKryo)

      try {
        body
      } finally {
        sc.stop()
        System.clearProperty("spark.driver.port")
        System.clearProperty("spark.hostPort")
      }
    }
  }

  /**
    * Perform a very basic sanity check on the model
    *
    * @param path path to the model coefficients file
    * @param expectedNumCoefficients expected number of non-zero coefficients
    * @return true if the model is sane
    */
  def modelSane(path: Path, expectedNumCoefficients: Int): Boolean = {
    val modelAvro = AvroIOUtils.readFromSingleAvro[BayesianLinearModelAvro](
      sc, path.toString, BayesianLinearModelAvro.getClassSchema.toString)

    val means = modelAvro.head.getMeans()
    means.filter(x => x.getValue != 0).size == expectedNumCoefficients
  }

  def modelContainsIntercept(path: Path): Boolean = {
    val modelAvro = AvroIOUtils.readFromSingleAvro[BayesianLinearModelAvro](
      sc, path.toString, BayesianLinearModelAvro.getClassSchema.toString)

    modelAvro.head.getMeans().map(nameTermValueAvro =>
      NameAndTerm(nameTermValueAvro.getName().toString, nameTermValueAvro.getTerm().toString)
    ).toSet.contains(NameAndTerm.INTERCEPT_NAME_AND_TERM)
  }

  /**
    * Evaluate the model by calculating RMSE between its scores and the validation set
    *
    * @param driver the driver instance used for training
    * @param modelPath base path to the GAME model files
    * @return validation error for the model
    */
  def evaluateModel(driver: Driver, modelPath: Path): Double = {
    val featureShardIdToFeatureMapMap = driver.prepareFeatureMaps()
    val gameDataSet = driver.prepareGameDataSet(featureShardIdToFeatureMapMap)

    val gameModel = ModelProcessingUtils.loadGameModelFromHDFS(
      featureShardIdToFeatureMapMap, modelPath.toString, sc)

    val (_, evaluator) = driver.prepareValidatingEvaluator(
      driver.params.validateDirsOpt.get, featureShardIdToFeatureMapMap)

    val scores = gameModel.score(gameDataSet).scores
    evaluator.evaluate(scores)
  }

  /**
    * Run the Game driver with the specified arguments
    *
    * @param args the command-line arguments
    */
  def runDriver(args: Array[String]) = {
    val params = Params.parseFromCommandLine(args)
    val logger = new PhotonLogger(s"${params.outputDir}/log", sc)
    val driver = new Driver(params, sc, logger)

    driver.run()
    logger.close()
    driver
  }
}

object DriverTest {
  val fs = FileSystems.getDefault
  val inputPath = getClass.getClassLoader.getResource("GameIntegTest/input").getPath
  val trainPath = inputPath + "/train"
  val testPath = inputPath + "/test"
  val featurePath = inputPath + "/feature-lists"
  val numIterations = 1
  val numExecutors = 1
  val numPartitionsForFixedEffectDataSet = numExecutors * 2
  val numPartitionsForRandomEffectDataSet = numExecutors * 2
  val tol = 1e-5

  /**
    * Default arguments to the Game driver
    */
  def defaultArgs = Map(
    "task-type" -> TaskType.LINEAR_REGRESSION.toString,
    "train-input-dirs" -> trainPath,
    "validate-input-dirs" -> testPath,
    "feature-name-and-term-set-path" -> featurePath,
    "num-iterations" -> numIterations.toString,
    "num-output-files-for-random-effect-model" -> "-1")

  /**
    * Fixed effect arguments with serious optimization. It's useful when we care about the model performance
    */
  def fixedEffectSeriousRunArgs = defaultArgs ++ Map(
    "feature-shard-id-to-feature-section-keys-map" -> "shard1:features",
    "updating-sequence" -> "global",

    // fixed-effect optimization config
    "fixed-effect-optimization-configurations" ->
      "global:10,1e-5,10,1,tron,l2",

    // fixed-effect data config
    "fixed-effect-data-configurations" ->
      s"global:shard1,$numPartitionsForFixedEffectDataSet")

  /**
   * Fixed effect arguments with "toy" optimization. It's useful when we don't care about the model performance
   */
  def fixedEffectToyRunArgs = defaultArgs ++ Map(
    "feature-shard-id-to-feature-section-keys-map" -> "shard1:features",
    "updating-sequence" -> "global",

    // fixed-effect optimization config
    "fixed-effect-optimization-configurations" ->
        "global:1,1e-5,10,1,tron,l2",

    // fixed-effect data config
    "fixed-effect-data-configurations" ->
        s"global:shard1,$numPartitionsForFixedEffectDataSet")

  /**
   * Random effect arguments with "serious" optimization. It's useful when we care about the model performance
   */
  def randomEffectSeriousRunArgs = {
    val userRandomEffectRegularizationWeight = 1
    val songRandomEffectRegularizationWeight = 1

    defaultArgs ++ Map(
      "feature-shard-id-to-feature-section-keys-map" ->
          "shard2:userFeatures|shard3:songFeatures",
      "updating-sequence" -> "per-user,per-song,per-artist",

      // random-effect optimization config
      "random-effect-optimization-configurations" ->
          (s"per-user:10,1e-5,$userRandomEffectRegularizationWeight,1,tron,l2|" +
              s"per-song:10,1e-5,$songRandomEffectRegularizationWeight,1,tron,l2|" +
              s"per-artist:10,1e-5,$userRandomEffectRegularizationWeight,1,tron,l2"),

      // random-effect data config
      "random-effect-data-configurations" ->
          (s"per-user:userId,shard2,$numPartitionsForRandomEffectDataSet,-1,0,-1,index_map|" +
              s"per-song:songId,shard3,$numPartitionsForRandomEffectDataSet,-1,0,-1,index_map|" +
              s"per-artist:artistId,shard3,$numPartitionsForRandomEffectDataSet,-1,0,-1,RANDOM=2"))
  }

  /**
   * Random effect arguments with "toy" optimization. It's useful when we don't care about the model performance
   */
  def randomEffectToyRunArgs = {
    val userRandomEffectRegularizationWeight = 1
    val songRandomEffectRegularizationWeight = 1

    defaultArgs ++ Map(
      "feature-shard-id-to-feature-section-keys-map" ->
          "shard2:userFeatures|shard3:songFeatures",
      "updating-sequence" -> "per-user,per-song,per-artist",

      // random-effect optimization config
      "random-effect-optimization-configurations" ->
          (s"per-user:1,1e-5,$userRandomEffectRegularizationWeight,1,tron,l2|" +
              s"per-song:1,1e-5,$songRandomEffectRegularizationWeight,1,tron,l2|" +
              s"per-artist:1,1e-5,$userRandomEffectRegularizationWeight,1,tron,l2"),

      // random-effect data config
      "random-effect-data-configurations" ->
          (s"per-user:userId,shard2,$numPartitionsForRandomEffectDataSet,-1,0,-1,index_map|" +
              s"per-song:songId,shard3,$numPartitionsForRandomEffectDataSet,-1,0,-1,index_map|" +
              s"per-artist:artistId,shard3,$numPartitionsForRandomEffectDataSet,-1,0,-1,RANDOM=2"))
  }

  /**
   * Fixed and random effect arguments. It's useful when we care about the model performance
   */
  def fixedAndRandomEffectSeriousRunArgs = {
    fixedEffectSeriousRunArgs ++ randomEffectSeriousRunArgs ++ Map(
      "feature-shard-id-to-feature-section-keys-map" ->
          "shard1:features,userFeatures,songFeatures|shard2:features,userFeatures|shard3:songFeatures",
      "updating-sequence" -> "global,per-user,per-song,per-artist"
    )
  }

  /**
    * Fixed and random effect arguments. It's useful when we don't care about the model performance
    */
  def fixedAndRandomEffectToyRunArgs = {
    fixedEffectToyRunArgs ++ randomEffectToyRunArgs ++ Map(
      "feature-shard-id-to-feature-section-keys-map" ->
        "shard1:features,userFeatures,songFeatures|shard2:features,userFeatures|shard3:songFeatures",
      "updating-sequence" -> "global,per-user,per-song,per-artist"
    )
  }

  /**
    * Build the path to the model coefficients file, given some model properties
    *
    * @param outputDir output base directory
    * @param outputMode output mode (best or all)
    * @param modelType model type (e.g. "fixed-effect", "random-effect")
    * @param modelName the model name
    * @return full path to model coefficients file
    */
  private def modelPath(outputDir: String, outputMode: String, modelType: String, modelName: String): Path =
    fs.getPath(outputDir, outputMode, modelType, modelName, "coefficients", "part-00000.avro")

  /**
    * Build the path to the model coefficients file
    *
    * @param outputDir output base directory
    * @param modelType model type (e.g. "fixed-effect", "random-effect")
    * @param modelName the model name
    * @return full path to model coefficients file
    */
  def allModelPath(outputDir: String, modelType: String, modelName: String): Path =
    modelPath(outputDir, "all/0", modelType, modelName)

  /**
    * Build the path to the best model coefficients file
    *
    * @param outputDir output base directory
    * @param modelType model type (e.g. "fixed-effect", "random-effect")
    * @param modelName the model name
    * @return full path to model coefficients file
    */
  def bestModelPath(outputDir: String, modelType: String, modelName: String): Path =
    modelPath(outputDir, "best", modelType, modelName)
}
