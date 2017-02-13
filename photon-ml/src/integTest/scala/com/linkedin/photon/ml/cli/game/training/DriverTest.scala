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

import java.nio.file.{FileSystems, Files, Path}

import org.apache.spark.SparkException

import org.testng.Assert._
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.{GameTestUtils, TaskType}
import com.linkedin.photon.ml.avro.AvroIOUtils
import com.linkedin.photon.ml.avro.data.NameAndTerm
import com.linkedin.photon.ml.avro.generated.BayesianLinearModelAvro
import com.linkedin.photon.ml.avro.model.ModelProcessingUtils
import com.linkedin.photon.ml.data.{AvroDataReader, GameConverters}
import com.linkedin.photon.ml.estimators.GameParams
import com.linkedin.photon.ml.evaluation._
import com.linkedin.photon.ml.io.ModelOutputMode
import com.linkedin.photon.ml.optimization.OptimizerType
import com.linkedin.photon.ml.optimization.OptimizerType.OptimizerType
import com.linkedin.photon.ml.test.{CommonTestUtils, SparkTestUtils, TestTemplateWithTmpDir}
import com.linkedin.photon.ml.util.{LongHashPartitioner, PhotonLogger, Utils}


import scala.collection.JavaConversions._


/**
 * Test cases for the GAME training driver
 */
class DriverTest extends SparkTestUtils with GameTestUtils with TestTemplateWithTmpDir {

  import CommonTestUtils._
  import DriverTest._

  @Test
  def testFixedEffectsWithIntercept(): Unit = sparkTest("testFixedEffectsWithIntercept", useKryo = true) {
    val outputDir = s"$getTmpDir/fixedEffects"
    // This is a baseline RMSE capture from an assumed-correct implementation on 4/14/2016
    val errorThreshold = 1.7
    val driver = runDriver(argArray(fixedEffectSeriousRunArgs() ++ Map("output-dir" -> outputDir)))
    val allFixedEffectModelPath = allModelPath(outputDir, "fixed-effect", "global")
    val bestFixedEffectModelPath = bestModelPath(outputDir, "fixed-effect", "global")

    println(allFixedEffectModelPath)
    assertTrue(Files.exists(allFixedEffectModelPath))
    assertTrue(Files.exists(bestFixedEffectModelPath))
    assertModelSane(allFixedEffectModelPath, expectedNumCoefficients = 14983)
    assertModelSane(bestFixedEffectModelPath, expectedNumCoefficients = 14983)
    assertTrue(evaluateModel(driver, fs.getPath(outputDir, "all/0")).head < errorThreshold)
    assertTrue(evaluateModel(driver, fs.getPath(outputDir, "best")).head < errorThreshold)
    assertTrue(modelContainsIntercept(allFixedEffectModelPath))
    assertTrue(modelContainsIntercept(bestFixedEffectModelPath))
  }

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def failedTestRunWithOutputDirExists(): Unit = sparkTest("failedTestRunWithOutputDirExists") {
    val outputDir = getTmpDir + "/failedTestRunWithOutputDirExists"
    Utils.createHDFSDir(outputDir, sc.hadoopConfiguration)
    runDriver(argArray(fixedEffectToyRunArgs() ++ Map("output-dir" -> outputDir)))
  }

  @Test
  def successfulTestRunWithOutputDirExists(): Unit = sparkTest("successfulTestRunWithOutputDirExists") {
    val outputDir = getTmpDir + "/successfulTestRunWithOutputDirExists"
    Utils.createHDFSDir(outputDir, sc.hadoopConfiguration)
    runDriver(argArray(fixedEffectToyRunArgs() ++
        Map("output-dir" -> outputDir, "delete-output-dir-if-exists" -> "true")))
  }

  @Test
  def testFixedEffectsWithoutIntercept(): Unit = sparkTest("testFixedEffectsWithoutIntercept", useKryo = true) {
    val outputDir = s"$getTmpDir/fixedEffects"
    runDriver(argArray(fixedEffectToyRunArgs() ++ Map("feature-shard-id-to-intercept-map" -> "shard1:false")
        ++ Map("output-dir" -> outputDir)))
    val allFixedEffectModelPath = allModelPath(outputDir, "fixed-effect", "global")
    val bestFixedEffectModelPath = bestModelPath(outputDir, "fixed-effect", "global")

    assertTrue(Files.exists(allFixedEffectModelPath))
    assertTrue(Files.exists(bestFixedEffectModelPath))
    assertModelSane(allFixedEffectModelPath, expectedNumCoefficients = 11597)
    assertModelSane(bestFixedEffectModelPath, expectedNumCoefficients = 11597)
    assertFalse(modelContainsIntercept(allFixedEffectModelPath))
    assertFalse(modelContainsIntercept(bestFixedEffectModelPath))
  }

  @Test
  def testSaveBestOnly(): Unit = sparkTest("saveBestOnly", useKryo = true) {
    val outputDir = s"$getTmpDir/fixedEffects"
    runDriver(argArray(fixedEffectToyRunArgs() ++ Map(
      "output-dir" -> outputDir,
      "model-output-mode" -> ModelOutputMode.BEST.toString)))
    val allFixedEffectModelPath = allModelPath(outputDir, "fixed-effect", "global")
    val bestFixedEffectModelPath = bestModelPath(outputDir, "fixed-effect", "global")

    assertFalse(Files.exists(allFixedEffectModelPath))
    assertTrue(Files.exists(bestFixedEffectModelPath))
  }

  @Test
  def testSaveNone(): Unit = sparkTest("saveNone", useKryo = true) {
    val outputDir = s"$getTmpDir/fixedEffects"
    runDriver(argArray(fixedEffectToyRunArgs() ++ Map(
      "output-dir" -> outputDir,
      "model-output-mode" -> ModelOutputMode.NONE.toString)))
    val allFixedEffectModelPath = allModelPath(outputDir, "fixed-effect", "global")
    val bestFixedEffectModelPath = bestModelPath(outputDir, "fixed-effect", "global")

    assertFalse(Files.exists(allFixedEffectModelPath))
    assertFalse(Files.exists(bestFixedEffectModelPath))
  }

  @Test
  def testRandomEffectsWithIntercept(): Unit = sparkTest("testRandomEffectsWithIntercept", useKryo = true) {
    val outputDir = s"$getTmpDir/randomEffects"
    // This is a baseline RMSE capture from an assumed-correct implementation on 4/14/2016
    val errorThreshold = 2.34
    val driver = runDriver(argArray(randomEffectSeriousRunArgs() ++ Map("output-dir" -> outputDir)))
    val userModelPath = bestModelPath(outputDir, "random-effect", "per-user")
    val songModelPath = bestModelPath(outputDir, "random-effect", "per-song")
    val artistModelPath = bestModelPath(outputDir, "random-effect", "per-artist")

    assertTrue(Files.exists(userModelPath))
    assertModelSane(userModelPath, expectedNumCoefficients = 21)
    assertTrue(modelContainsIntercept(userModelPath))

    assertTrue(Files.exists(songModelPath))
    assertModelSane(songModelPath, expectedNumCoefficients = 21)
    assertTrue(modelContainsIntercept(songModelPath))

    assertTrue(Files.exists(artistModelPath))
    assertModelSane(artistModelPath, expectedNumCoefficients = 21)
    assertTrue(modelContainsIntercept(artistModelPath))

    assertTrue(evaluateModel(driver, fs.getPath(outputDir, "best")).head < errorThreshold)
  }

  @Test
  def testRandomEffectsWithoutAnyIntercept(): Unit = sparkTest("testRandomEffectsWithoutAnyIntercept", useKryo = true) {
    val outputDir = s"$getTmpDir/randomEffects"
    runDriver(argArray(randomEffectToyRunArgs() ++
        Map("feature-shard-id-to-intercept-map" -> "shard2:false|shard3:false") ++ Map("output-dir" -> outputDir)))
    val userModelPath = bestModelPath(outputDir, "random-effect", "per-user")
    val songModelPath = bestModelPath(outputDir, "random-effect", "per-song")
    val artistModelPath = bestModelPath(outputDir, "random-effect", "per-artist")

    assertTrue(Files.exists(userModelPath))
    assertModelSane(userModelPath, expectedNumCoefficients = 20)
    assertFalse(modelContainsIntercept(userModelPath))

    assertTrue(Files.exists(songModelPath))
    assertModelSane(songModelPath, expectedNumCoefficients = 20)
    assertFalse(modelContainsIntercept(songModelPath))

    assertTrue(Files.exists(artistModelPath))
    assertModelSane(artistModelPath, expectedNumCoefficients = 20)
    assertFalse(modelContainsIntercept(artistModelPath))
  }

  @Test
  def testRandomEffectsWithPartialIntercept(): Unit
    = sparkTest("testRandomEffectsWithPartialIntercept", useKryo = true) {

      val outputDir = s"$getTmpDir/randomEffects"
      runDriver(argArray(randomEffectToyRunArgs() ++
          Map("feature-shard-id-to-intercept-map" -> "shard2:false|shard3:true") ++ Map("output-dir" -> outputDir)))
      val userModelPath = bestModelPath(outputDir, "random-effect", "per-user")
      val songModelPath = bestModelPath(outputDir, "random-effect", "per-song")
      val artistModelPath = bestModelPath(outputDir, "random-effect", "per-artist")

      assertTrue(Files.exists(userModelPath))
      assertModelSane(userModelPath, expectedNumCoefficients = 20)
      assertFalse(modelContainsIntercept(userModelPath))

      assertTrue(Files.exists(songModelPath))
      assertModelSane(songModelPath, expectedNumCoefficients = 21)
      assertTrue(modelContainsIntercept(songModelPath))

      assertTrue(Files.exists(artistModelPath))
      assertModelSane(artistModelPath, expectedNumCoefficients = 21)
      assertTrue(modelContainsIntercept(artistModelPath))
  }

  @Test
  def testFixedAndRandomEffects(): Unit = sparkTest("fixedAndRandomEffects", useKryo = true) {
    val outputDir = s"$getTmpDir/fixedAndRandomEffects"

    // This is a baseline RMSE capture from an assumed-correct implementation on 4/14/2016
    val errorThreshold = 2.2

    val driver = runDriver(argArray(fixedAndRandomEffectSeriousRunArgs() ++ Map("output-dir" -> outputDir)))

    val fixedEffectModelPath = bestModelPath(outputDir, "fixed-effect", "global")
    val userModelPath = bestModelPath(outputDir, "random-effect", "per-user")
    val songModelPath = bestModelPath(outputDir, "random-effect", "per-song")
    val artistModelPath = bestModelPath(outputDir, "random-effect", "per-artist")

    assertTrue(Files.exists(fixedEffectModelPath))
    assertModelSane(fixedEffectModelPath, expectedNumCoefficients = 15019)
    assertTrue(modelContainsIntercept(fixedEffectModelPath))

    assertTrue(Files.exists(userModelPath))
    assertModelSane(userModelPath, expectedNumCoefficients = 29, modelId = Some("1436929"))
    assertTrue(modelContainsIntercept(userModelPath))

    assertTrue(Files.exists(songModelPath))
    assertModelSane(songModelPath, expectedNumCoefficients = 21)
    assertTrue(modelContainsIntercept(songModelPath))

    assertTrue(Files.exists(artistModelPath))
    assertModelSane(artistModelPath, expectedNumCoefficients = 21)
    assertTrue(modelContainsIntercept(artistModelPath))

    assertTrue(evaluateModel(driver, fs.getPath(outputDir, "best")).head < errorThreshold)
  }

  @Test
  def testMultipleOptimizerConfigs(): Unit = sparkTest("multipleOptimizerConfigs", useKryo = true) {
    val outputDir = s"$getTmpDir/multipleOptimizerConfigs"

    // This is a baseline RMSE capture from an assumed-correct implementation on 4/14/2016
    val errorThreshold = 1.7

    val driver = runDriver(argArray(fixedEffectSeriousRunArgs() ++ Map(
      "output-dir" -> outputDir,
      "fixed-effect-optimization-configurations" ->
        ("global:10,1e-5,10,1,tron,l2;" +
          "global:10,1e-5,10,1,lbfgs,l2"))))

    val fixedEffectModelPath = bestModelPath(outputDir, "fixed-effect", "global")

    assertTrue(Files.exists(fixedEffectModelPath))
    assertModelSane(fixedEffectModelPath, expectedNumCoefficients = 14983)
    assertTrue(evaluateModel(driver, fs.getPath(outputDir, "best")).head < errorThreshold)
  }

  @DataProvider
  def shardedEvaluatorOfUnknownIdTypeProvider(): Array[Array[Any]] = {
    Array(
      Array(Seq(AUC, ShardedAUC("foo"))),
      Array(Seq(ShardedAUC("foo"), ShardedPrecisionAtK(10, "bar"))),
      Array(Seq(ShardedPrecisionAtK(1, "foo")))
    )
  }

  @Test(expectedExceptions = Array(classOf[SparkException]), dataProvider = "shardedEvaluatorOfUnknownIdTypeProvider")
  def evaluateFullModelWithShardedEvaluatorOfUnknownIdType(evaluatorTypes: Seq[EvaluatorType])
  : Unit = sparkTest("evaluateFullModelWithShardedEvaluatorOfUnknownIdType") {

    val outputDir = s"$getTmpDir/evaluateFullModelWithPrecisionAtKOfUnknownId"
    runDriver(argArray(fixedAndRandomEffectToyRunArgs() ++ Map(
      "output-dir" -> outputDir,
      EvaluatorType.cmdArgument -> evaluatorTypes.map(_.name).mkString(","))))
  }

  @Test
  def testNoValidatingDir(): Unit = sparkTest("noValidatingDir", useKryo = true) {
    val outputDir = s"$getTmpDir/testNoValidatingDir"

    // Verify that the system still works if we don't specify a validating dir
    runDriver(argArray(fixedEffectSeriousRunArgs() ++ Map("output-dir" -> outputDir) - "validate-input-dirs"))

    val fixedEffectModelPath = modelPath(outputDir, "all/0", "fixed-effect", "global")

    assertTrue(Files.exists(fixedEffectModelPath))
  }

  @Test
  def testOffHeapIndexMap(): Unit = sparkTest("offHeapIndexMap", useKryo = true) {
    val outputDir = s"$getTmpDir/offHeapIndexMap"

    // This is a baseline RMSE capture from an assumed-correct implementation on 4/14/2016
    val errorThreshold = 2.2

    val indexMapPath = getClass.getClassLoader.getResource("GameIntegTest/input/feature-indexes").getPath
    val driver = runDriver(argArray(
      fixedAndRandomEffectSeriousRunArgs() ++ Map(
        "output-dir" -> outputDir,
        "offheap-indexmap-dir" -> indexMapPath,
        "offheap-indexmap-num-partitions" -> "1")))

    val fixedEffectModelPath = bestModelPath(outputDir, "fixed-effect", "global")
    val userModelPath = bestModelPath(outputDir, "random-effect", "per-user")
    val songModelPath = bestModelPath(outputDir, "random-effect", "per-song")
    val artistModelPath = bestModelPath(outputDir, "random-effect", "per-artist")

    assertTrue(Files.exists(fixedEffectModelPath))
    assertModelSane(fixedEffectModelPath, expectedNumCoefficients = 15032)
    assertTrue(modelContainsIntercept(fixedEffectModelPath))

    assertTrue(Files.exists(userModelPath))
    assertModelSane(userModelPath, expectedNumCoefficients = 39, modelId = Some("1436929"))
    assertTrue(modelContainsIntercept(userModelPath))

    assertTrue(Files.exists(songModelPath))
    assertModelSane(songModelPath, expectedNumCoefficients = 31)
    assertTrue(modelContainsIntercept(songModelPath))

    assertTrue(Files.exists(artistModelPath))
    assertModelSane(artistModelPath, expectedNumCoefficients = 31)
    assertTrue(modelContainsIntercept(artistModelPath))

    assertTrue(evaluateModel(driver, fs.getPath(outputDir, "best")).head < errorThreshold)
  }

  /**
    * Perform a very basic sanity check on the model
    *
    * @param path path to the model coefficients file
    * @param expectedNumCoefficients expected number of non-zero coefficients
    * @return true if the model is sane
    */
  def assertModelSane(path: Path, expectedNumCoefficients: Int, modelId: Option[String] = None): Unit = {
    val modelAvro = AvroIOUtils.readFromSingleAvro[BayesianLinearModelAvro](
      sc, path.toString, BayesianLinearModelAvro.getClassSchema.toString)

    val model = modelId match {
      case Some(id) =>
        val m = modelAvro.find { m => m.getModelId.toString == id }
        assertTrue(m.isDefined, s"Model id $id not found.")
        m.get

      case _ => modelAvro.head
    }

    assertEquals(model.getMeans.count(x => x.getValue != 0), expectedNumCoefficients)
  }

  def modelContainsIntercept(path: Path): Boolean = {

    val modelAvro = AvroIOUtils.readFromSingleAvro[BayesianLinearModelAvro](
      sc, path.toString, BayesianLinearModelAvro.getClassSchema.toString)

    modelAvro.head.getMeans.map(
      nameTermValueAvro => NameAndTerm(nameTermValueAvro.getName.toString, nameTermValueAvro.getTerm.toString)
    ).toSet.contains(NameAndTerm.INTERCEPT_NAME_AND_TERM)
  }

  /**
    * Evaluate the model by the specified evaluators with the validation data set
    *
    * @param driver the driver instance used for training
    * @param modelPath base path to the GAME model files
    * @return evaluation results for each specified evaluator
    */
  def evaluateModel(driver: Driver, modelPath: Path): Seq[Double] = {
    val idTypeSet = Set("userId", "artistId", "songId")
    val indexMapLoaders = driver.prepareFeatureMaps()
    val featureSectionMap = driver.params.featureShardIdToFeatureSectionKeysMap
    val dr = new AvroDataReader(sc)
    val testData = dr.readMerged(testPath, indexMapLoaders, featureSectionMap, 2)

    val partitioner = new LongHashPartitioner(testData.rdd.partitions.length)
    val gameDataSet = GameConverters.getGameDataSetFromDataFrame(
      testData,
      featureSectionMap.keys.toSet,
      idTypeSet,
      isResponseRequired = true)
      .partitionBy(partitioner)

    val validatingLabelsAndOffsetsAndWeights = gameDataSet
      .mapValues(gameData => (gameData.response, gameData.offset, gameData.weight))
    validatingLabelsAndOffsetsAndWeights.count()

    val evaluator = new RMSEEvaluator(validatingLabelsAndOffsetsAndWeights)
    val (gameModel, _) = ModelProcessingUtils.loadGameModelFromHDFS(Some(indexMapLoaders), modelPath.toString, sc)

    val scores = gameModel.score(gameDataSet).scores
    Seq(evaluator.evaluate(scores))
  }

  /**
    * Run the Game driver with the specified arguments
    *
    * @param args the command-line arguments
    */
  def runDriver(args: Array[String]): Driver = {
    val params = GameParams.parseFromCommandLine(args)
    val logger = new PhotonLogger(s"${params.outputDir}/log", sc)
    val driver = new Driver(params, sc, logger)

    driver.run()
    logger.close()
    driver
  }
}

object DriverTest {

  private val fs = FileSystems.getDefault
  private val inputPath = getClass.getClassLoader.getResource("GameIntegTest/input").getPath
  private val trainPath = inputPath + "/train"
  private val testPath = inputPath + "/test"
  private val featurePath = inputPath + "/feature-lists"
  private val numIterations = 1
  private val numExecutors = 1
  private val numPartitionsForFixedEffectDataSet = numExecutors * 2
  private val numPartitionsForRandomEffectDataSet = numExecutors * 2

  /**
   * Default arguments to the Game driver
   */
  def defaultArgs: Map[String, String] = Map(
    "task-type" -> TaskType.LINEAR_REGRESSION.toString,
    "train-input-dirs" -> trainPath,
    "validate-input-dirs" -> testPath,
    "feature-name-and-term-set-path" -> featurePath,
    "num-iterations" -> numIterations.toString,
    "num-output-files-for-random-effect-model" -> "-1")

  /**
   * Fixed effect arguments with serious optimization. It's useful when we care about the model performance
   */
  def fixedEffectSeriousRunArgs(optType: OptimizerType = OptimizerType.TRON): Map[String, String] = defaultArgs ++ Map(
    "feature-shard-id-to-feature-section-keys-map" -> "shard1:features",
    "updating-sequence" -> "global",

    // fixed-effect optimization config
    "fixed-effect-optimization-configurations" ->
      s"global:10,1e-5,10,1,$optType,l2",

    // fixed-effect data config
    "fixed-effect-data-configurations" ->
      s"global:shard1,$numPartitionsForFixedEffectDataSet")

  /**
   * Fixed effect arguments with "toy" optimization. It's useful when we don't care about the model performance
   */
  def fixedEffectToyRunArgs(optType: OptimizerType = OptimizerType.TRON): Map[String, String] = defaultArgs ++ Map(

    "feature-shard-id-to-feature-section-keys-map" -> "shard1:features",
    "updating-sequence" -> "global",

    // fixed-effect optimization config
    "fixed-effect-optimization-configurations" ->
        s"global:1,1e-5,10,1,$optType,l2",

    // fixed-effect data config
    "fixed-effect-data-configurations" ->
        s"global:shard1,$numPartitionsForFixedEffectDataSet")

  /**
   * Random effect arguments with "serious" optimization. It's useful when we care about the model performance
   */
  def randomEffectSeriousRunArgs(optType: OptimizerType = OptimizerType.TRON): Map[String, String] = {
    val userRandomEffectRegularizationWeight = 1
    val songRandomEffectRegularizationWeight = 1

    defaultArgs ++ Map(
      "feature-shard-id-to-feature-section-keys-map" ->
          "shard2:userFeatures|shard3:songFeatures",
      "updating-sequence" -> "per-user,per-song,per-artist",

      // random-effect optimization config
      "random-effect-optimization-configurations" ->
          (s"per-user:10,1e-5,$userRandomEffectRegularizationWeight,1,$optType,l2|" +
              s"per-song:10,1e-5,$songRandomEffectRegularizationWeight,1,$optType,l2|" +
              s"per-artist:10,1e-5,$userRandomEffectRegularizationWeight,1,$optType,l2"),

      // random-effect data config
      "random-effect-data-configurations" ->
          (s"per-user:userId,shard2,$numPartitionsForRandomEffectDataSet,-1,0,-1,index_map|" +
              s"per-song:songId,shard3,$numPartitionsForRandomEffectDataSet,-1,0,-1,index_map|" +
              s"per-artist:artistId,shard3,$numPartitionsForRandomEffectDataSet,-1,0,-1,RANDOM=2"))
  }

  /**
   * Random effect arguments with "toy" optimization. It's useful when we don't care about the model performance
   */
  def randomEffectToyRunArgs(optType: OptimizerType = OptimizerType.TRON): Map[String, String] = {
    val userRandomEffectRegularizationWeight = 1
    val songRandomEffectRegularizationWeight = 1

    defaultArgs ++ Map(
      "feature-shard-id-to-feature-section-keys-map" ->
          "shard2:userFeatures|shard3:songFeatures",
      "updating-sequence" -> "per-user,per-song,per-artist",

      // random-effect optimization config
      "random-effect-optimization-configurations" ->
          (s"per-user:1,1e-5,$userRandomEffectRegularizationWeight,1,$optType,l2|" +
              s"per-song:1,1e-5,$songRandomEffectRegularizationWeight,1,$optType,l2|" +
              s"per-artist:1,1e-5,$userRandomEffectRegularizationWeight,1,$optType,l2"),

      // random-effect data config
      "random-effect-data-configurations" ->
          (s"per-user:userId,shard2,$numPartitionsForRandomEffectDataSet,-1,0,-1,index_map|" +
              s"per-song:songId,shard3,$numPartitionsForRandomEffectDataSet,-1,0,-1,index_map|" +
              s"per-artist:artistId,shard3,$numPartitionsForRandomEffectDataSet,-1,0,-1,RANDOM=2"))
  }

  /**
   * Fixed and random effect arguments. It's useful when we care about the model performance
   */
  def fixedAndRandomEffectSeriousRunArgs(optType: OptimizerType = OptimizerType.TRON): Map[String, String] = {
    fixedEffectSeriousRunArgs(optType) ++ randomEffectSeriousRunArgs(optType) ++ Map(
      "feature-shard-id-to-feature-section-keys-map" ->
          "shard1:features,userFeatures,songFeatures|shard2:features,userFeatures|shard3:songFeatures",
      "updating-sequence" -> "global,per-user,per-song,per-artist"
    )
  }

  /**
    * Fixed and random effect arguments. It's useful when we don't care about the model performance
    */
  def fixedAndRandomEffectToyRunArgs(optType: OptimizerType = OptimizerType.TRON): Map[String, String] = {
    fixedEffectToyRunArgs(optType) ++ randomEffectToyRunArgs(optType) ++ Map(
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
