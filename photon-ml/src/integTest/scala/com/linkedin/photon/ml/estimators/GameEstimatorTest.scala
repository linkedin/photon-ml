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
package com.linkedin.photon.ml.estimators

import org.apache.spark.sql.DataFrame

import org.testng.Assert.{assertEquals, fail}
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.GameTestUtils
import com.linkedin.photon.ml.TaskType
import com.linkedin.photon.ml.TaskType.TaskType
import com.linkedin.photon.ml.avro.data.NameAndTermFeatureSetContainer
import com.linkedin.photon.ml.data._
import com.linkedin.photon.ml.evaluation._
import com.linkedin.photon.ml.test.SparkTestUtils
import com.linkedin.photon.ml.util._

class GameEstimatorTest extends SparkTestUtils with GameTestUtils {
  import GameEstimatorTest._

  @Test
  def testPrepareFixedEffectTrainingDataSet(): Unit = sparkTest("prepareFixedEffectTrainingDataSet") {
    val featureSectionMap = fixedEffectOnlyFeatureSectionMap
    val data = getData(trainPath, featureSectionMap)

    val partitioner = new LongHashPartitioner(data.rdd.partitions.length)
    val gameDataSet = GameConverters.getGameDataSetFromDataFrame(
      data,
      featureSectionMap.keys.toSet,
      idTypeSet,
      isResponseRequired = true)
      .partitionBy(partitioner)

    val params = fixedEffectOnlyParams
    val estimator = getEstimator(params)
    val trainingDataSet = estimator.prepareTrainingDataSet(gameDataSet)

    assertEquals(trainingDataSet.size, 1)

    trainingDataSet("global") match {
      case ds: FixedEffectDataSet =>
        assertEquals(ds.labeledPoints.count(), 34810)
        assertEquals(ds.numFeatures, 30045)

      case _ => fail("Wrong dataset type.")
    }
  }

  @Test
  def testPrepareFixedAndRandomEffectTrainingDataSet(): Unit =
    sparkTest("prepareFixedAndRandomEffectTrainingDataSet", useKryo = true) {
      val featureSectionMap = fixedAndRandomEffectFeatureSectionMap
      val data = getData(trainPath, featureSectionMap)

      val partitioner = new LongHashPartitioner(data.rdd.partitions.length)
      val gameDataSet = GameConverters.getGameDataSetFromDataFrame(
        data,
        featureSectionMap.keys.toSet,
        idTypeSet,
        isResponseRequired = true)
        .partitionBy(partitioner)

      val params = fixedAndRandomEffectParams
      val estimator = getEstimator(params)
      val trainingDataSet = estimator.prepareTrainingDataSet(gameDataSet)

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

  @DataProvider
  def multipleEvaluatorTypeProvider(): Array[Array[Any]] = {
    Array(
      Array(Seq(RMSE, SquaredLoss)),
      Array(Seq(LogisticLoss, AUC, ShardedPrecisionAtK(1, "userId"), ShardedPrecisionAtK(10, "songId"))),
      Array(Seq(AUC, ShardedAUC("userId"), ShardedAUC("songId"))),
      Array(Seq(PoissonLoss))
    )
  }

  @Test(dataProvider = "multipleEvaluatorTypeProvider")
  def testMultipleEvaluatorsWithFixedEffectModel(
      evaluatorTypes: Seq[EvaluatorType]): Unit = sparkTest("testMultipleEvaluatorsWithFixedEffect", useKryo = true) {

    val featureSectionMap = fixedEffectOnlyFeatureSectionMap
    val data = getData(testPath, featureSectionMap)

    val params = fixedEffectOnlyParams
    params.evaluatorTypes = evaluatorTypes

    val estimator = getEstimator(params)

    val (_, evaluators) = estimator.prepareValidatingEvaluators(data)
    evaluators
      .zip(evaluatorTypes)
      .foreach { case (evaluator, evaluatorType) => assertEquals(evaluator.getEvaluatorName, evaluatorType.name) }
  }

  @Test(dataProvider = "multipleEvaluatorTypeProvider")
  def testMultipleEvaluatorsWithFullModel(
      evaluatorTypes: Seq[EvaluatorType]): Unit = sparkTest("testMultipleEvaluatorsWithFullModel", useKryo = true) {

    val featureSectionMap = fixedAndRandomEffectFeatureSectionMap
    val data = getData(testPath, featureSectionMap)

    val params = fixedAndRandomEffectParams
    params.evaluatorTypes = evaluatorTypes

    val estimator = getEstimator(params)

    val (_, evaluators) = estimator.prepareValidatingEvaluators(data)
    evaluators
      .zip(evaluatorTypes)
      .foreach { case (evaluator, evaluatorType) => assertEquals(evaluator.getEvaluatorName, evaluatorType.name) }
  }

  @DataProvider
  def taskAndDefaultEvaluatorTypeProvider(): Array[Array[Any]] = {
    Array(
      Array(TaskType.LINEAR_REGRESSION, RMSE),
      Array(TaskType.LOGISTIC_REGRESSION, AUC),
      Array(TaskType.SMOOTHED_HINGE_LOSS_LINEAR_SVM, AUC),
      Array(TaskType.POISSON_REGRESSION, PoissonLoss)
    )
  }

  @Test(dataProvider = "taskAndDefaultEvaluatorTypeProvider")
  def testDefaultEvaluator(
      taskType: TaskType,
      defaultEvaluatorType: EvaluatorType): Unit = sparkTest("testDefaultEvaluator", useKryo = true) {

    val featureSectionMap = fixedEffectOnlyFeatureSectionMap
    val data = getData(testPath, featureSectionMap)

    val params = fixedEffectOnlyParams
    params.taskType = taskType

    val estimator = getEstimator(params)

    val (_, evaluators) = estimator.prepareValidatingEvaluators(data)
    assertEquals(evaluators.head.getEvaluatorName, defaultEvaluatorType.name)
  }

  /**
   * Returns the feature map loaders for these test cases.
   *
   * @param featureSectionMap the map pairing destination feature bags to source feature sections
   * @return initialized feature map loaders
   */
  def getFeatureMapLoaders(featureSectionMap: Map[String, Set[String]]): Map[String, DefaultIndexMapLoader] = {
    val featureSectionKeySet = featureSectionMap.values.flatten.toSet
    val nameAndTermFeatureSetContainer = NameAndTermFeatureSetContainer.readNameAndTermFeatureSetContainerFromTextFiles(
      featurePath, featureSectionKeySet, sc.hadoopConfiguration)

    featureSectionMap.map { case (shardId, featureSectionKeys) =>
      val featureMap = nameAndTermFeatureSetContainer
        .getFeatureNameAndTermToIndexMap(featureSectionKeys, isAddingIntercept = true)
        .map { case (k, v) => Utils.getFeatureKey(k.name, k.term) -> v }
        .toMap

      val indexMapLoader = new DefaultIndexMapLoader(featureMap)
      indexMapLoader.prepare(sc, null, shardId)
      (shardId, indexMapLoader)
    }
  }

  /**
   * Returns test case data frame
   *
   * @param inputPath path to the data file(s)
   * @param featureSectionMap the map pairing destination feature bags to source feature sections
   * @return loaded data frame
   */
  def getData(inputPath: String, featureSectionMap: Map[String, Set[String]]): DataFrame = {
    val featureMapLoaders = getFeatureMapLoaders(featureSectionMap)
    val dr = new AvroDataReader(sc)
    dr.readMerged(inputPath, featureMapLoaders, featureSectionMap, 2)
  }

  /**
   * Creates a test estimator from the params
   *
   * @param params game params object specifying estimator parameters
   * @return the created estimator
   */
  def getEstimator(params: GameParams): GameEstimator = {
    val logger = new PhotonLogger(s"${params.outputDir}/log", sc)
    new GameEstimator(params, sc, logger)
  }
}

object GameEstimatorTest {

  private val inputPath: String = getClass.getClassLoader.getResource("GameIntegTest/input").getPath
  private val trainPath: String = inputPath + "/train"
  private val testPath: String = inputPath + "/test"
  private val featurePath: String = inputPath + "/feature-lists"
  private val tol = 1e-5
  private val idTypeSet = Set("userId", "artistId", "songId")

  /**
   * Section map for fixed effect only models
   */
  private val fixedEffectOnlyFeatureSectionMap = Map("shard1" -> Set("features"))

  /**
   * Section map for fixed and random effect models
   */
  private val fixedAndRandomEffectFeatureSectionMap = Map(
      "shard1" -> Set("features", "userFeatures", "songFeatures"),
      "shard2" -> Set("features", "userFeatures"),
      "shard3" -> Set("songFeatures"))

  /**
   * Default estimator params for tests on fixed-effect-only models
   */
  def fixedEffectOnlyParams: GameParams = {
    val params = new GameParams
    params.fixedEffectDataConfigurations = Map(
      "global" -> FixedEffectDataConfiguration.parseAndBuildFromString("shard1,2"))

    params
  }

  /**
    * Default estimator params for tests on fixed and random effect models
    */
  def fixedAndRandomEffectParams: GameParams = {
    val params = new GameParams
    params.fixedEffectDataConfigurations = Map(
      "global" -> FixedEffectDataConfiguration.parseAndBuildFromString("shard1,2"))

    params.randomEffectDataConfigurations = Map(
      "per-user" -> RandomEffectDataConfiguration.parseAndBuildFromString("userId,shard2,2,-1,0,-1,index_map"),
      "per-song" -> RandomEffectDataConfiguration.parseAndBuildFromString("songId,shard3,2,-1,0,-1,index_map"),
      "per-artist" -> RandomEffectDataConfiguration.parseAndBuildFromString("artistId,shard3,2,-1,0,-1,RANDOM=2"))

    params
  }
}
