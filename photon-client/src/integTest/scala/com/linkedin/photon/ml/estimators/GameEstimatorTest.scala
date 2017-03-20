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
package com.linkedin.photon.ml.estimators

import breeze.linalg.DenseVector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.testng.Assert.{assertEquals, assertNotEquals, fail}
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.TaskType
import com.linkedin.photon.ml.TaskType.TaskType
import com.linkedin.photon.ml.Types._
import com.linkedin.photon.ml.constants.MathConst
import com.linkedin.photon.ml.data._
import com.linkedin.photon.ml.data.avro.{AvroDataReader, NameAndTermFeatureSetContainer}
import com.linkedin.photon.ml.evaluation.Evaluator.EvaluationResults
import com.linkedin.photon.ml.evaluation.EvaluatorType._
import com.linkedin.photon.ml.evaluation.{EvaluatorType, ShardedAUC, ShardedPrecisionAtK}
import com.linkedin.photon.ml.model.{FixedEffectModel, GAMEModel}
import com.linkedin.photon.ml.normalization.{NormalizationContext, NormalizationType}
import com.linkedin.photon.ml.stat.BasicStatisticalSummary
import com.linkedin.photon.ml.test.{CommonTestUtils, SparkTestUtils}
import com.linkedin.photon.ml.util._

/**
 * Integration tests for GAMEEstimator.
 *
 * The test data set here is a subset of the Yahoo! music data set available on the internet.
 */
class GameEstimatorTest extends SparkTestUtils with GameTestUtils {

  import GameEstimatorTest._

  /**
   * A very simple test that fits a toy data set using only the GAMEEstimator (not the full Driver).
   * This is useful to understand the minimum setting in which a GAMEEstimator will function properly,
   * and to verify the algorithms manually.
   *
   * @note
   *   Intercepts are optional in GAMEEstimator, but GAMEDriver will setup an intercept by default if
   *   none is specified in GAMEParams.featureShardIdToInterceptMap.
   *   This happens in GAMEDriver.prepareFeatureMapsDefault, and there only.
   *   Here, we have to setup an intercept manually, otherwise GAMEEstimator learns only a dependence on the
   *   features.
   */
  @Test(dataProvider = "trivialLabeledPoints")
  def simpleHardcodedTest(labeledPoints: Seq[LabeledPoint]): Unit = sparkTest("simpleHardcodedTest") {

    // This example has only a single fixed effect
    val (coordinateId, featureShardId) = ("global", "features")
    val nDimensions = labeledPoints.head.features.size

    // Setup feature names a feature index from feature name to feature index
    // Have to drop in an intercept "feature" but intercepts are turned ON by default
    val featureNames = (0 until nDimensions).map(i => s"feature-$i").toSet
    val featureIndexMap = DefaultIndexMap(featureNames) + ((Constants.INTERCEPT_NAME_TERM, nDimensions))

    // Generate a Spark DataFrame containing labeled points (label, x, y)
    // :+ 1.0 is because we need to add that column to make up a correct design matrix, including that 1.0
    // to multiply by the intercept term.
    val label = GameConverters.FieldNames.RESPONSE
    val trainingData: DataFrame = new SQLContext(sc)
      .createDataFrame(labeledPoints
        .map { point: LabeledPoint =>
          val x = Vectors.dense(point.features.toDenseVector.toArray :+ 1.0).toSparse
          (point.label, x) })
      .toDF(label, featureShardId)

    // Calling rdd explicitly here to avoid a typed encoder lookup in Spark 2.1
    val stats = BasicStatisticalSummary(trainingData.select(featureShardId).rdd.map(_.getAs[SparkVector](0)))

    // We set args for the GAMEEstimator - only: so this is the minimum set of params required by GAMEEstimator
    // Default number of passes over the coordinates (numIterations) is 1, which is all we need if
    // we have a single fixed effect model
    val args = Map[String, String](
      "task-type" -> TaskType.LINEAR_REGRESSION.toString,
      "feature-shard-id-to-feature-section-keys-map" -> s"$featureShardId:${featureNames.mkString(",")}",
      "fixed-effect-data-configurations" -> s"$coordinateId:$featureShardId,1",
      "fixed-effect-optimization-configurations" -> s"$coordinateId:100,1e-11,0.3,1,LBFGS,l2",
      "updating-sequence" -> coordinateId,
      "train-input-dirs" -> "", // required by GAMEParams parser, but not used in GAMEEstimator
      "validate-input-dirs" -> "", // required by GAMEParams parser, but not used in GAMEEstimator
      "output-dir" -> "", // required by GAMEParams parser, but not used in GAMEEstimator
      "feature-name-and-term-set-path" -> "")  // required by GAMEParams parser, but not used in GAMEEstimator
    val params = GameParams.parseFromCommandLine(CommonTestUtils.argArray(args))

    // Compute normalization contexts based on statistics of the training data for this (unique) feature shard
    val normalizationContexts: Option[Map[String, NormalizationContext]] =
      Some(Map((featureShardId, stats))
        .mapValues { featureShardStats =>
          val intercept: Option[Int] = featureIndexMap.get(Constants.INTERCEPT_NAME_TERM)
          NormalizationContext(params.normalizationType, featureShardStats, intercept)
        })

    // Create GAMEEstimator and fit model
    // Returns (model, evaluation, optimizer config)
    val models: Seq[(GAMEModel, Option[EvaluationResults], String)] =
        createEstimator(params, "simpleTest")._1.fit(trainingData, validationData = None, normalizationContexts)

    val model = models.head._1.getModel(coordinateId).head.asInstanceOf[FixedEffectModel].model

    // Reference values from scikit-learn
    assertEquals(model.coefficients.means(0), 0.3215554473500486, 1e-12)
    assertEquals(model.coefficients.means(1), 0.17904355431985355, 1e-12)
    assertEquals(model.coefficients.means(2), 0.4122241763914806, 1e-12)
  }

  /**
   * Build GAMEParams for each possible value of the normalization.
   *
   * @return An collection of GAMEParams with one entry for each normalization type.
   */
  @DataProvider
  def normalizationParameters(): Array[Array[GameParams]] =

    Array(NormalizationType.values.map {
      normalizationType =>
        GameParams.parseFromCommandLine(CommonTestUtils.argArray(Map[String, String](
          "task-type" -> TaskType.LINEAR_REGRESSION.toString,
          "feature-shard-id-to-feature-section-keys-map" -> s"features:feature-0,feature-1",
          "fixed-effect-data-configurations" -> s"global:features,1",
          "fixed-effect-optimization-configurations" -> s"global:100,1e-11,0,1,LBFGS,l2",
          "updating-sequence" -> "global",
          "normalization-type" -> normalizationType.toString,
          "train-input-dirs" -> "",
          "validate-input-dirs" -> "",
          "output-dir" -> "",
          "feature-name-and-term-set-path" -> ""
        )))
    })

  /**
   * In this test, we just verify that we can fit a model with each possible normalization type, without verifying that
   * the results are correct.
   *
   * Not using DataProviders, because they can't be composed to have multiple DataProviders serving arguments to
   * a single function.
   *
   * @note when regularization is not involved, the model should be invariant under renormalization, since we undo
   *       normalization after training.
   */
  @Test
  def testNormalization(): Unit = sparkTest("testNormalization") {

    val answer = DenseVector(0.34945501725815586, 0.26339479490270173, 0.4366125400310442)

    for (params: GameParams <- normalizationParameters()(0)) {

      // This example has only a single fixed effect
      val (coordinateId, featureShardId) = ("global", "features")
      val labeledPoints: Seq[LabeledPoint] = trivialLabeledPoints()(0)(0)
      val nDimensions = labeledPoints.head.features.size

      // Setup feature names a feature index from feature name to feature index
      // Have to drop in an intercept "feature" but intercepts are turned ON by default
      val featureNames = (0 until nDimensions).map(i => s"feature-$i").toSet
      val featureIndexMap = DefaultIndexMap(featureNames) + ((Constants.INTERCEPT_NAME_TERM, nDimensions))

      // Generate a Spark DataFrame containing labeled points (label, x, y)
      // :+ 1.0 is because we need to add that column to make up a correct design matrix, including that 1.0
      // to multiply by the intercept term.
      val label = GameConverters.FieldNames.RESPONSE
      val trainingData: DataFrame = new SQLContext(sc)
        .createDataFrame(labeledPoints
          .map { point: LabeledPoint =>
            val x = Vectors.dense(point.features.toDenseVector.toArray :+ 1.0).toSparse
            (point.label, x)
          })
        .toDF(label, featureShardId)

      // Calling rdd explicitly here to avoid a typed encoder lookup in Spark 2.1
      val stats = BasicStatisticalSummary(trainingData.select(featureShardId).rdd.map(_.getAs[SparkVector](0)))

      // Compute normalization contexts based on statistics of the training data for this (unique) feature shard
      val normalizationContexts: Option[Map[String, NormalizationContext]] =
        Some(Map((featureShardId, stats))
          .mapValues { featureShardStats =>
            val intercept: Option[Int] = featureIndexMap.get(Constants.INTERCEPT_NAME_TERM)
            NormalizationContext(params.normalizationType, featureShardStats, intercept)
          })

      // Create GAMEEstimator and fit model
      // Returns (model, evaluation, optimizer config)
      val models: Seq[(GAMEModel, Option[EvaluationResults], String)] =
      createEstimator(params, "simpleTest")._1.fit(trainingData, validationData = None, normalizationContexts)

      val model = models.head._1.getModel(coordinateId).head.asInstanceOf[FixedEffectModel].model
      for (i <- 0 until nDimensions)
        assertNotEquals(model.coefficients.means(i), 0.0, 1e-12, s"${params.normalizationType.toString} i")
      answer.toArray.zip(model.coefficients.means.toArray).foreach {
        case (exp, act) => assertEquals(exp, act, MathConst.MEDIUM_PRECISION_TOLERANCE_THRESHOLD)
      }
    }
  }

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
    val (estimator, _) = createEstimator(params, "prepareFixedEffectTrainingDataSet")
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
      val (estimator, _) = createEstimator(params, "prepareFixedAndRandomEffectTrainingDataSet")
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
  def multipleEvaluatorTypeProvider(): Array[Array[Any]] =

    Array(
      Array(Seq(RMSE, SquaredLoss)),
      Array(Seq(LogisticLoss, AUC, ShardedPrecisionAtK(1, "userId"), ShardedPrecisionAtK(10, "songId"))),
      Array(Seq(AUC, ShardedAUC("userId"), ShardedAUC("songId"))),
      Array(Seq(PoissonLoss))
    )

  @Test(dataProvider = "multipleEvaluatorTypeProvider")
  def testMultipleEvaluatorsWithFixedEffectModel(
      evaluatorTypes: Seq[EvaluatorType]): Unit = sparkTest("multipleEvaluatorsWithFixedEffect", useKryo = true) {

    val featureSectionMap = fixedEffectOnlyFeatureSectionMap
    val data = getData(testPath, featureSectionMap)

    val params = fixedEffectOnlyParams
    params.evaluatorTypes = evaluatorTypes

    val (estimator, _) = createEstimator(params, "multipleEvaluatorTypeProvider")

    val (_, evaluators) = estimator.prepareValidationEvaluators(data)
    evaluators
      .zip(evaluatorTypes)
      .foreach { case (evaluator, evaluatorType) => assertEquals(evaluator.getEvaluatorName, evaluatorType.name) }
  }

  @Test(dataProvider = "multipleEvaluatorTypeProvider")
  def testMultipleEvaluatorsWithFullModel(
      evaluatorTypes: Seq[EvaluatorType]): Unit = sparkTest("multipleEvaluatorsWithFullModel", useKryo = true) {

    val featureSectionMap = fixedAndRandomEffectFeatureSectionMap
    val data = getData(testPath, featureSectionMap)

    val params = fixedAndRandomEffectParams
    params.evaluatorTypes = evaluatorTypes

    val (estimator, _) = createEstimator(params, "multipleEvaluatorsWithFullModel")

    val (_, evaluators) = estimator.prepareValidationEvaluators(data)
    evaluators
      .zip(evaluatorTypes)
      .foreach { case (evaluator, evaluatorType) => assertEquals(evaluator.getEvaluatorName, evaluatorType.name) }
  }

  @DataProvider
  def taskAndDefaultEvaluatorTypeProvider(): Array[Array[Any]] =

    Array(
      Array(TaskType.LINEAR_REGRESSION, RMSE),
      Array(TaskType.LOGISTIC_REGRESSION, AUC),
      Array(TaskType.SMOOTHED_HINGE_LOSS_LINEAR_SVM, AUC),
      Array(TaskType.POISSON_REGRESSION, PoissonLoss)
    )

  @Test(dataProvider = "taskAndDefaultEvaluatorTypeProvider")
  def testDefaultEvaluator(
      taskType: TaskType,
      defaultEvaluatorType: EvaluatorType): Unit = sparkTest("taskAndDefaultEvaluatorTypeProvider", useKryo = true) {

    val featureSectionMap = fixedEffectOnlyFeatureSectionMap
    val data = getData(testPath, featureSectionMap)

    val params = fixedEffectOnlyParams
    params.taskType = taskType

    val (estimator, _) = createEstimator(params, "taskAndDefaultEvaluatorTypeProvider")

    val (_, evaluators) = estimator.prepareValidationEvaluators(data)
    assertEquals(evaluators.head.getEvaluatorName, defaultEvaluatorType.name)
  }

  /**
   * Returns the feature map loaders for these test cases.
   *
   * @param featureSectionMap The map pairing destination feature bags to source feature sections
   * @return Initialized feature map loaders
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

      (shardId, new DefaultIndexMapLoader(sc, featureMap))
    }
  }

  /**
   * Returns test case data frame.
   *
   * @param inputPath Path to the data file(s)
   * @param featureSectionMap The map pairing destination feature bags to source feature sections
   * @return Loaded data frame
   */
  def getData(inputPath: String, featureSectionMap: Map[String, Set[String]]): DataFrame = {

    val featureMapLoaders = getFeatureMapLoaders(featureSectionMap)
    val dr = new AvroDataReader(sc)
    dr.readMerged(inputPath, featureMapLoaders, featureSectionMap, 2)
  }

  /**
   * Creates a test estimator from the params.
   *
   * @param params GAME params object specifying estimator parameters
   * @param testName Optional name of the test: if provided the logs with go to that dir in tmp dir
   *                 (tmp dir is per thread)
   * @return The created estimator and the logger (so we can use it in tests if needed)
   */
  def createEstimator(params: GameParams, testName: String = "GenericTest"): (GameEstimator, PhotonLogger) = {

    val logFile = s"$getTmpDir/$testName"
    val logger = new PhotonLogger(logFile, sc)
    val estimator = new GameEstimator(sc, params, logger)
    (estimator, logger)
  }
}

object GameEstimatorTest {

  // The test data set here is a subset of the Yahoo! music data set available on the internet.
  private val inputPath = getClass.getClassLoader.getResource("GameIntegTest/input").getPath
  private val trainPath = inputPath + "/train"
  private val testPath = inputPath + "/test"
  private val featurePath = inputPath + "/feature-lists"
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
   * Default estimator params for tests on fixed-effect-only models.
   *
   * @return params to train fixed effect model
   */
  private def fixedEffectOnlyParams: GameParams = {

    val params = new GameParams
    params.fixedEffectDataConfigurations = Map(
      "global" -> FixedEffectDataConfiguration.parseAndBuildFromString("shard1,2"))
    params
  }

  /**
   * Default estimator params for tests on fixed and random effect models.
   *
   * @return param to train fixed and random effect models
   */
  private def fixedAndRandomEffectParams: GameParams = {

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
