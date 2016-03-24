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
import java.nio.file.{Files, FileSystems, Path}
import org.apache.spark.{SparkConf, SparkContext}
import org.testng.annotations.{BeforeMethod, AfterMethod, DataProvider, Test}
import org.testng.Assert._

import com.linkedin.photon.ml.avro.AvroIOUtils
import com.linkedin.photon.ml.avro.generated.BayesianLinearModelAvro
import com.linkedin.photon.ml.data.{FixedEffectDataSet, RandomEffectDataSet}
import com.linkedin.photon.ml.SparkContextConfiguration
import com.linkedin.photon.ml.supervised.TaskType
import com.linkedin.photon.ml.supervised.TaskType.TaskType
import com.linkedin.photon.ml.test.{CommonTestUtils, SparkTestUtils, TestTemplateWithTmpDir}
import com.linkedin.photon.ml.util.PhotonLogger

class DriverIntegTest extends SparkTestUtils with TestTemplateWithTmpDir {
  import DriverIntegTest._
  import CommonTestUtils._

  @Test
  def testFixedEffects = sparkTest("fixedEffects", useKryo = true) {
    val outputDir = s"$getTmpDir/fixedEffects"

    runDriver(argArray(fixedEffectArgs ++ Map(
      "output-dir" -> outputDir)))

    val fixedEffectModelPath = modelPath(outputDir, "fixed-effect", "shard1")

    assertTrue(Files.exists(fixedEffectModelPath))
    assertTrue(modelSane(fixedEffectModelPath, expectedNumCoefficients = 14983))
  }

  @Test
  def testRandomEffects = sparkTest("randomEffects", useKryo = true) {
    val outputDir = s"$getTmpDir/randomEffects"

    runDriver(argArray(randomEffectArgs ++ Map(
      "output-dir" -> outputDir)))

    val userModelPath = modelPath(outputDir, "random-effect", "userId-shard2")
    val songModelPath = modelPath(outputDir, "random-effect", "songId-shard3")
    val artistModelPath = modelPath(outputDir, "random-effect", "artistId-shard3")

    assertTrue(Files.exists(userModelPath))
    assertTrue(modelSane(userModelPath, expectedNumCoefficients = 21))

    assertTrue(Files.exists(songModelPath))
    assertTrue(modelSane(songModelPath, expectedNumCoefficients = 21))

    assertTrue(Files.exists(artistModelPath))
    assertTrue(modelSane(artistModelPath, expectedNumCoefficients = 21))
  }

  @Test
  def testFixedAndRandomEffects = sparkTest("fixedAndRandomEffects", useKryo = true) {
    val outputDir = s"$getTmpDir/fixedAndRandomEffects"

    runDriver(argArray(fixedAndRandomEffectArgs ++ Map(
      "output-dir" -> outputDir)))

    val fixedEffectModelPath = modelPath(outputDir, "fixed-effect", "shard1")
    val userModelPath = modelPath(outputDir, "random-effect", "userId-shard2")
    val songModelPath = modelPath(outputDir, "random-effect", "songId-shard3")
    val artistModelPath = modelPath(outputDir, "random-effect", "artistId-shard3")

    assertTrue(Files.exists(fixedEffectModelPath))
    assertTrue(modelSane(fixedEffectModelPath, expectedNumCoefficients = 15017))

    assertTrue(Files.exists(userModelPath))
    assertTrue(modelSane(userModelPath, expectedNumCoefficients = 29))

    assertTrue(Files.exists(songModelPath))
    assertTrue(modelSane(songModelPath, expectedNumCoefficients = 21))

    assertTrue(Files.exists(artistModelPath))
    assertTrue(modelSane(artistModelPath, expectedNumCoefficients = 21))
  }

  @Test
  def testPrepareFixedEffectTrainingDataSet = sparkTest("prepareFixedEffectTrainingDataSet", useKryo = true) {
    val outputDir = s"$getTmpDir/prepareFixedEffectTrainingDataSet"

    val args = argArray(fixedEffectArgs ++ Map(
      "output-dir" -> outputDir))

    val driver = new Driver(
      Params.parseFromCommandLine(args), sc, new PhotonLogger(s"$outputDir/log", sc))

    val featureShardIdToFeatureMapMap = driver.prepareFeatureMaps()
    val gameDataSet = driver.prepareGameDataSet(featureShardIdToFeatureMapMap)
    val trainingDataSet = driver.prepareTrainingDataSet(gameDataSet)

    assertEquals(trainingDataSet.size, 1)

    val fixedEffectDataSet = trainingDataSet("global") match {
      case ds: FixedEffectDataSet =>
        assertEquals(ds.labeledPoints.count, 34810)
        assertEquals(ds.numFeatures, 30045)

      case _ => fail("Wrong dataset type.")
    }
  }

  @Test
  def testPrepareFixedAndRandomEffectTrainingDataSet =
      sparkTest("prepareFixedAndRandomEffectTrainingDataSet", useKryo = true) {
    val outputDir = s"$getTmpDir/prepareFixedEffectTrainingDataSet"

    val args = argArray(fixedAndRandomEffectArgs ++ Map(
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
        assertEquals(ds.labeledPoints.count, 34810)
        assertEquals(ds.numFeatures, 30085)

      case _ => fail("Wrong dataset type.")
    }

    // per-user data
    trainingDataSet("per-user") match {
      case ds: RandomEffectDataSet =>
        assertEquals(ds.activeData.count, 33110)

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
        assertEquals(ds.activeData.count, 23167)

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
        assertEquals(ds.activeData.count, 4471)

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
  def testMultipleOptimizerConfigs = sparkTest("multipleOptimizerConfigs", useKryo = true) {
    val outputDir = s"$getTmpDir/multipleOptimizerConfigs"

    runDriver(argArray(fixedEffectArgs ++ Map(
      "output-dir" -> outputDir,
      "fixed-effect-optimization-configurations" ->
        ("global:10,1e-5,10,1,tron,l2;" +
         "global:10,1e-5,10,1,lbfgs,l2"))))

    val fixedEffectModelPath = modelPath(outputDir, "fixed-effect", "shard1")

    assertTrue(Files.exists(fixedEffectModelPath))
    assertTrue(modelSane(fixedEffectModelPath, expectedNumCoefficients = 14982))
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
        sc.stop
        System.clearProperty("spark.driver.port")
        System.clearProperty("spark.hostPort")
      }
    }
  }

  /**
   * Perform a very basic sanity check on the model
   *
   * @param path path to the model coefficients file
   * @return true if the model is sane
   */
  def modelSane(path: Path, expectedNumCoefficients: Int): Boolean = {
    val modelAvro = AvroIOUtils.readFromSingleAvro[BayesianLinearModelAvro](
      sc, path.toString, BayesianLinearModelAvro.getClassSchema.toString)

    val means = modelAvro(0).getMeans()
    means.filter(x => x.getValue != 0).size == expectedNumCoefficients
  }

  /**
   * Run the Game driver with the specified arguments
   *
   * @param args the command-line arguments
   */
  def runDriver(args: Array[String]) {
    val params = Params.parseFromCommandLine(args)
    val logger = new PhotonLogger(s"${params.outputDir}/log", sc)
    val driver = new Driver(params, sc, logger)

    driver.run
    logger.close
  }
}

object DriverIntegTest {
  val fs = FileSystems.getDefault()
  val inputPath = getClass.getClassLoader.getResource("GameDriverIntegTest/input").getPath
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
    "save-models-to-hdfs" -> true.toString)

  /**
   * Default fixed effect arguments
   */
  def fixedEffectArgs = defaultArgs ++ Map(
    "feature-shard-id-to-feature-section-keys-map" -> "shard1:features",
    "updating-sequence" -> "global",

    // fixed-effect optimization config
    "fixed-effect-optimization-configurations" ->
      "global:10,1e-5,10,1,tron,l2",

    // fixed-effect data config
    "fixed-effect-data-configurations" ->
      s"global:shard1,$numPartitionsForFixedEffectDataSet")

  /**
   * Default random effect arguments
   */
  def randomEffectArgs = {
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
   * Default fixed and random effect arguments
   */
  def fixedAndRandomEffectArgs = {
    fixedEffectArgs ++ randomEffectArgs ++ Map(
      "feature-shard-id-to-feature-section-keys-map" ->
        "shard1:features,userFeatures,songFeatures|shard2:features,userFeatures|shard3:songFeatures",
      "updating-sequence" -> "global,per-user,per-song,per-artist"
    )
  }

  /**
   * Build the path to the model coefficients file, given some model properties
   *
   * @param outputDir output base directory
   * @param modelType model type (e.g. "fixed-effect", "random-effect")
   * @param shardId the shard id designator
   * @return full path to model coefficients file
   */
  def modelPath(outputDir: String, modelType: String, shardId: String): Path = fs.getPath(
      outputDir, "best", modelType, shardId, "coefficients", "part-00000.avro")
}
