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
package com.linkedin.photon.ml.data.avro

import java.io.File

import scala.collection.JavaConversions._
import scala.collection.immutable.IndexedSeq

import org.apache.avro.file.DataFileReader
import org.apache.avro.specific.SpecificDatumReader
import org.apache.hadoop.fs.Path
import org.testng.Assert._
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.avro.generated.FeatureSummarizationResultAvro
import com.linkedin.photon.ml.cli.game.training.GameTrainingDriver
import com.linkedin.photon.ml.constants.StorageLevel
import com.linkedin.photon.ml.estimators.GameEstimator
import com.linkedin.photon.ml.index.{DefaultIndexMap, DefaultIndexMapLoader, IndexMap, IndexMapLoader}
import com.linkedin.photon.ml.model._
import com.linkedin.photon.ml.optimization._
import com.linkedin.photon.ml.optimization.game.{FixedEffectOptimizationConfiguration, RandomEffectOptimizationConfiguration}
import com.linkedin.photon.ml.stat.BasicStatisticalSummary
import com.linkedin.photon.ml.supervised.classification.LogisticRegressionModel
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import com.linkedin.photon.ml.test.{SparkTestUtils, TestTemplateWithTmpDir}
import com.linkedin.photon.ml.util.VectorUtils.toSparseVector
import com.linkedin.photon.ml.util._
import com.linkedin.photon.ml.{Constants, TaskType}

/**
 * Unit tests for model processing utilities.
 */
class ModelProcessingUtilsIntegTest extends SparkTestUtils with TestTemplateWithTmpDir {

  import ModelProcessingUtilsIntegTest._

  /**
   * Generate a decent GAME model for subsequent tests.
   * This GAME model is a logistic regression. It has 1 fixed effect, and 2 different random effect models,
   * the random effect models have 2 and 3 "items" respectively.
   *
   * Notes:
   * =====
   * - for the features, we have two feature spaces, one for the fix model, and one for the random model
   *   Each model has its own, separate feature space, but feature values can be shared between spaces.
   *   Features shared between spaces have a unique name, but possibly different indices.
   * - we cheat a little bit by giving the feature spaces the same name as the models,
   *   because it makes testing easier, esp. in the case where the model is loaded without specified
   *   index IndexMapLoaders: in that case, the feature space names are the model names.
   *
   * On HDFS the files for this model are:
   *
   *   hdfs://hostname:port/tmp/GAMELaserModelTest/GAMEModel/fixed-effect/fixed/coefficients/part-00000.avro
   *   hdfs://hostname:port/tmp/GAMELaserModelTest/GAMEModel/fixed-effect/fixed/id-info
   *   hdfs://hostname:port/tmp/GAMELaserModelTest/GAMEModel/random-effect/RE1/coefficients/_SUCCESS
   *   hdfs://hostname:port/tmp/GAMELaserModelTest/GAMEModel/random-effect/RE1/coefficients/part-00000.avro
   *   hdfs://hostname:port/tmp/GAMELaserModelTest/GAMEModel/random-effect/RE1/coefficients/part-00001.avro
   *   hdfs://hostname:port/tmp/GAMELaserModelTest/GAMEModel/random-effect/RE1/id-info
   *   hdfs://hostname:port/tmp/GAMELaserModelTest/GAMEModel/random-effect/RE2/coefficients/_SUCCESS
   *   hdfs://hostname:port/tmp/GAMELaserModelTest/GAMEModel/random-effect/RE2/coefficients/part-00000.avro
   *   hdfs://hostname:port/tmp/GAMELaserModelTest/GAMEModel/random-effect/RE2/coefficients/part-00001.avro
   *   hdfs://hostname:port/tmp/GAMELaserModelTest/GAMEModel/random-effect/RE2/id-info
   *
   * @return ([[GameModel]], feature index loaders, feature names)
   */
  def makeGameModel(): (GameModel, Map[String, DefaultIndexMapLoader], Map[String, IndexedSeq[String]]) = {

    val numFeatures = Map("fixed" -> 10, "RE1" -> 10, "RE2" -> 10)

    val featureNames =
      numFeatures
        .mapValues { nf => (0 until nf).map(i => Utils.getFeatureKey("n" + i, "t")) }

    val featureIndexLoaders =
      featureNames
        .map { case (modelType, modelFeatures) => (modelType, DefaultIndexMapLoader(sc, modelFeatures)) }
        .map(identity) // .map(identity) needed because of: https://issues.scala-lang.org/browse/SI-7005

    // Fixed effect model
    val glm = new LogisticRegressionModel(Coefficients(numFeatures("fixed"))(1,2,5)(11,21,51))
    val fixedEffectModel = new FixedEffectModel(sc.broadcast(glm), "fixed")

    val glmRE11 = LogisticRegressionModel(Coefficients(numFeatures("RE1"))(1, 5, 7)(111, 511, 911))
    val glmRE12 = LogisticRegressionModel(Coefficients(numFeatures("RE1"))(1, 2)(112, 512))
    val glmRE1RDD = sc.parallelize(List(("RE1Item1", glmRE11), ("RE1Item2", glmRE12)))
    val RE1Model = new RandomEffectModel(glmRE1RDD, "randomEffectModel1", "RE1")

    val glmRE21 = LogisticRegressionModel(Coefficients(numFeatures("RE2"))(3, 4, 6)(321, 421, 621))
    val glmRE22 = LogisticRegressionModel(Coefficients(numFeatures("RE2"))(4, 5)(322, 422))
    val glmRE23 = LogisticRegressionModel(Coefficients(numFeatures("RE2"))(2, 7, 8)(323, 423, 523))
    val glmRE2RDD = sc.parallelize(List(("RE2Item1", glmRE21), ("RE2Item2", glmRE22), ("RE2Item3", glmRE23)))
    val RE2Model = new RandomEffectModel(glmRE2RDD, "randomEffectModel2", "RE2")

    (GameModel(("fixed", fixedEffectModel), ("RE1", RE1Model), ("RE2", RE2Model)),
      featureIndexLoaders,
      featureNames)
  }

  /**
   * Test that we can load a simple GAME model with fixed and random effects, given a feature index.
   */
  @Test
  def testLoadAndSaveGameModels(): Unit = sparkTest("testLoadAndSaveGameModels") {

    // Default number of output files
    val numberOfOutputFilesForRandomEffectModel = 2

    val (gameModel, featureIndexLoaders, _) = makeGameModel()
    val outputDir = new Path(getTmpDir)

    ModelProcessingUtils.saveGameModelToHDFS(
      sc,
      outputDir,
      gameModel,
      TaskType.LOGISTIC_REGRESSION,
      GAME_OPTIMIZATION_CONFIGURATION,
      Some(numberOfOutputFilesForRandomEffectModel),
      featureIndexLoaders,
      VectorUtils.DEFAULT_SPARSITY_THRESHOLD)

    val fs = outputDir.getFileSystem(sc.hadoopConfiguration)

    assertTrue(fs.exists(outputDir))

    // Check if the numberOfOutputFilesForRandomEffectModel parameter is working or not
    val randomEffectModelCoefficientsDir = new Path(
      outputDir,
      s"${AvroConstants.RANDOM_EFFECT}/RE1/${AvroConstants.COEFFICIENTS}")
    val numRandomEffectModelFiles = fs
      .listStatus(randomEffectModelCoefficientsDir)
      .count(_.getPath.toString.contains("part"))

    assertEquals(numRandomEffectModelFiles, numberOfOutputFilesForRandomEffectModel,
      s"Expected number of random effect model files: $numberOfOutputFilesForRandomEffectModel, " +
        s"found: $numRandomEffectModelFiles")

    // Check if the models loaded correctly and they are the same as the models saved previously
    // The second value returned is the feature index, which we don't need here
    val (loadedGameModel, _) = ModelProcessingUtils.loadGameModelFromHDFS(
      sc,
      outputDir,
      StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL,
      Some(featureIndexLoaders))
    assertTrue(gameModel == loadedGameModel)
  }

  /**
   * Test that we can save a GAME model with custom sparsity threshold
   */
  @Test
  def testSparsityThreshold(): Unit = sparkTest("testSparsityThreshold") {

    // Model sparsity threshold
    val modelSparsityThreshold = 12.0

    // Default number of output files
    val numberOfOutputFilesForRandomEffectModel = 2

    val (gameModel, featureIndexLoaders, _) = makeGameModel()
    val outputDir = new Path(getTmpDir)

    ModelProcessingUtils.saveGameModelToHDFS(
      sc,
      outputDir,
      gameModel,
      TaskType.LOGISTIC_REGRESSION,
      GAME_OPTIMIZATION_CONFIGURATION,
      Some(numberOfOutputFilesForRandomEffectModel),
      featureIndexLoaders,
      modelSparsityThreshold)

    val fs = outputDir.getFileSystem(sc.hadoopConfiguration)

    assertTrue(fs.exists(outputDir))

    val randomEffectModelCoefficientsDir = new Path(
      outputDir,
      s"${AvroConstants.RANDOM_EFFECT}/RE1/${AvroConstants.COEFFICIENTS}")
    val numRandomEffectModelFiles = fs
      .listStatus(randomEffectModelCoefficientsDir)
      .count(_.getPath.toString.contains("part"))

    assertEquals(numRandomEffectModelFiles, numberOfOutputFilesForRandomEffectModel,
      s"Expected number of random effect model files: $numberOfOutputFilesForRandomEffectModel, " +
        s"found: $numRandomEffectModelFiles")

    val (loadedGameModel, _) = ModelProcessingUtils.loadGameModelFromHDFS(
      sc,
      outputDir,
      StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL,
      Some(featureIndexLoaders))

    loadedGameModel.getModel("fixed") match {
      case Some(model: FixedEffectModel) =>
        assertEquals(
          model
            .modelBroadcast
            .value
            .coefficients
            .means
            .activeValuesIterator
            .toSet,
          Set(21, 51))

      case other => fail(s"Unexpected model: $other")
    }
  }

  /**
   * Test that we can load a model even if we don't have a feature index. In that case, the feature index is
   * generated as part of loading the model, and it will not directly match the feature index before the save.
   */
  @Test
  def testLoadGameModelsWithoutFeatureIndex(): Unit = sparkTest("testLoadGameModelsWithoutFeatureIndex") {

    val (gameModel, featureIndexLoaders, _) = makeGameModel()
    val outputDir = new Path(getTmpDir)

    ModelProcessingUtils.saveGameModelToHDFS(
      sc,
      outputDir,
      gameModel,
      TaskType.LOGISTIC_REGRESSION,
      GAME_OPTIMIZATION_CONFIGURATION,
      randomEffectModelFileLimit = None,
      featureIndexLoaders,
      VectorUtils.DEFAULT_SPARSITY_THRESHOLD)

    val (loadedGameModel, newFeatureIndexLoaders) = ModelProcessingUtils.loadGameModelFromHDFS(
      sc,
      outputDir,
      StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL,
      None)

    // Since the feature index after the load is not the same as before the save, we need to calculate an
    // invariant that survives the save/load. That invariant is Map[feature value, feature name].
    def features(glm: GeneralizedLinearModel, featureIndexLoader: IndexMapLoader): Array[(String, Double)] =
      ModelProcessingUtils.extractGLMFeatures(glm, featureIndexLoader.indexMapForRDD())

    // calling zip directly on maps doesn't zip on keys but just zips the underlying (key, value) iterables
    (gameModel.toSortedMap zip loadedGameModel.toSortedMap) foreach {

      case ((n1: String, m1: FixedEffectModel), (n2: String, m2: FixedEffectModel)) =>

        assertEquals(n1, n2)
        assertEquals(features(m1.model, featureIndexLoaders(n1)), features(m2.model, newFeatureIndexLoaders(n2)))

      case ((n1: String, m1: RandomEffectModel), (n2: String, m2: RandomEffectModel)) =>

        assertEquals(n1, n2)
        m1.modelsRDD.join(m2.modelsRDD).mapValues{ case (glm1, glm2) =>
          assertEquals(features(glm1, featureIndexLoaders(n1)), features(glm2, newFeatureIndexLoaders(n2)))
        }.collect

      case _ =>
    }
  }

  /**
   * Test that we can extract all features from a GAME model correctly.
   */
  @Test
  def testExtractGameModelFeatures(): Unit = sparkTest("testExtractGameModelFeatures") {

    val (gameModel, featureIndexLoaders, featureNames) = makeGameModel()
    val outputDir = new Path(getTmpDir)

    ModelProcessingUtils.saveGameModelToHDFS(
      sc,
      outputDir,
      gameModel,
      TaskType.LOGISTIC_REGRESSION,
      GAME_OPTIMIZATION_CONFIGURATION,
      randomEffectModelFileLimit = None,
      featureIndexLoaders,
      VectorUtils.DEFAULT_SPARSITY_THRESHOLD)

    // Check if the models loaded correctly and they are the same as the models saved previously
    // The first value returned is the feature index, which we don't need here
    val (loadedGameModel, newFeatureIndexLoaders) = ModelProcessingUtils.loadGameModelFromHDFS(
      sc,
      outputDir,
      StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL,
      None)

    // Let's extract all features from the GAME model...
    val features = ModelProcessingUtils.extractGameModelFeatures(sc, loadedGameModel, newFeatureIndexLoaders)

    // ... and verify the models
    features.foreach {
      case ((AvroConstants.FIXED_EFFECT, "fixed"), modelRDD) =>
        val calculated: Array[(String, Double)] = modelRDD.collect()(0)._2
        val ans = List(1, 2, 5).map(i => featureNames("fixed")(i)) zip List(11,21,51)
        assert(calculated sameElements ans)

      case ((AvroConstants.RANDOM_EFFECT, "RE1"), modelRDD) =>
        val features = featureNames("RE1")
        modelRDD.collect.foreach {

          case ("RE1Item1", coefficients) =>
            assert(coefficients sameElements (List(1, 5, 7).map(i => features(i)) zip List(111, 511, 911)))

          case ("RE1Item2", coefficients) =>
            assert(coefficients sameElements (List(1, 2).map(i => features(i)) zip List(112, 512)))
        }

      case ((AvroConstants.RANDOM_EFFECT, "RE2"), modelRDD) =>
        val features = featureNames("RE2")
        modelRDD.collect.foreach {

          case ("RE2Item1", coefficients) =>
            assert(coefficients sameElements (List(3, 4, 6).map(i => features(i)) zip List(321, 421, 621)))

          case ("RE2Item2", coefficients) =>
            assert(coefficients sameElements (List(4, 5).map(i => features(i)) zip List(322, 422)))

          case ("RE2Item3", coefficients) =>
            assert(coefficients sameElements (List(2, 7, 8).map(i => features(i)) zip List(323, 423, 523)))
        }
    }
  }

  @DataProvider
  def matrixFactorizationConfigProvider(): Array[Array[Any]] = {
    Array(
      Array(0, 0, 0),
      Array(1, 0, 0),
      Array(1, 1, 0),
      Array(1, 0, 1),
      Array(1, 1, 1),
      Array(5, 10, 10)
    )
  }

  /**
   * Test that we can save and load model metadata.
   *
   * TODO: this is incomplete - need to check that more parameters are loaded back correctly
   */
  @Test
  def testSaveAndLoadGameModelMetadata(): Unit = sparkTest("testSaveAndLoadGameModelMetadata") {

    val outputDir = new Path(getTmpDir)

    ModelProcessingUtils.saveGameModelMetadataToHDFS(sc, outputDir, TASK_TYPE, GAME_OPTIMIZATION_CONFIGURATION)

    assertEquals(
      TASK_TYPE,
      ModelProcessingUtils
        .loadGameModelMetadataFromHDFS(sc, outputDir)
        .getOrElse(GameTrainingDriver.trainingTask, TaskType.NONE))
  }

  /**
   * Test computing and writing out [[BasicStatisticalSummary]].
   */
  @Test
  def testWriteBasicStatistics(): Unit = sparkTest("testWriteBasicStatistics") {

    val dim: Int = 5
    val minVector = toSparseVector(Array((0, 1.5d), (1, 0d), (2, 0d), (3, 6.7d), (4, 2.33d)), dim)
    val maxVector = toSparseVector(Array((0, 10d), (1, 0d), (2, 0d), (3, 7d), (4, 4d)), dim)
    val normL1Vector = toSparseVector(Array((0, 1d), (1, 0d), (2, 0d), (3, 7d), (4, 4d)), dim)
    val normL2Vector = toSparseVector(Array((0, 2d), (1, 0d), (2, 0d), (3, 8d), (4, 5d)), dim)
    val numNonzeros = toSparseVector(Array((0, 6d), (1, 0d), (2, 0d), (3, 3d), (4, 89d)), dim)
    val meanVector = toSparseVector(Array((0, 1.1d), (3, 2.4d), (4, 3.6d)), dim)
    val varVector = toSparseVector(Array((0, 1d), (3, 7d), (4, 0.5d)), dim)

    val summary = BasicStatisticalSummary(
      mean = meanVector,
      variance = varVector,
      count = 101L,
      numNonzeros = numNonzeros,
      max = maxVector,
      min = minVector,
      normL1 = normL1Vector,
      normL2 = normL2Vector,
      meanAbs = meanVector)

    val indexMap: IndexMap = new DefaultIndexMap(Map(
      "f0" + Constants.DELIMITER -> 0,
      "f1" + Constants.DELIMITER + "t1" -> 1,
      "f2" + Constants.DELIMITER -> 2,
      "f3" + Constants.DELIMITER + "t3" -> 3,
      "f4" + Constants.DELIMITER -> 4))

    val tempOut = new Path(getTmpDir, "summary-output")
    ModelProcessingUtils.writeBasicStatistics(sc, summary, tempOut, indexMap)

    val reader = DataFileReader.openReader[FeatureSummarizationResultAvro](
      new File(tempOut.toString + "/part-00000.avro"),
      new SpecificDatumReader[FeatureSummarizationResultAvro]())
    var count = 0
    while (reader.hasNext) {
      val record = reader.next()
      val feature = record.getFeatureName + Constants.DELIMITER + record.getFeatureTerm
      val featureId = indexMap(feature)
      val metrics = record.getMetrics.map {case (key, value) => (String.valueOf(key), value)}
      var foundMatchedOne = true
      featureId match {
        case 0 =>
          assertEquals(feature, "f0" + Constants.DELIMITER)
          assertEquals(metrics("min"), 1.5d, EPSILON)
          assertEquals(metrics("max"), 10d, EPSILON)
          assertEquals(metrics("normL1"), 1d, EPSILON)
          assertEquals(metrics("normL2"), 2d, EPSILON)
          assertEquals(metrics("numNonzeros"), 6d, EPSILON)
          assertEquals(metrics("mean"), 1.1d, EPSILON)
          assertEquals(metrics("variance"), 1d, EPSILON)

        case 1 =>
          assertEquals(feature, "f1" + Constants.DELIMITER + "t1")
          assertEquals(metrics("min"), 0d, EPSILON)
          assertEquals(metrics("max"), 0d, EPSILON)
          assertEquals(metrics("normL1"), 0d, EPSILON)
          assertEquals(metrics("normL2"), 0d, EPSILON)
          assertEquals(metrics("numNonzeros"), 0d, EPSILON)
          assertEquals(metrics("mean"), 0d, EPSILON)
          assertEquals(metrics("variance"), 0d, EPSILON)

        case 2 =>
          assertEquals(feature, "f2" + Constants.DELIMITER)
          assertEquals(metrics("min"), 0d, EPSILON)
          assertEquals(metrics("max"), 0d, EPSILON)
          assertEquals(metrics("normL1"), 0d, EPSILON)
          assertEquals(metrics("normL2"), 0d, EPSILON)
          assertEquals(metrics("numNonzeros"), 0d, EPSILON)
          assertEquals(metrics("mean"), 0d, EPSILON)
          assertEquals(metrics("variance"), 0d, EPSILON)

        case 3 =>
          assertEquals(feature, "f3" + Constants.DELIMITER + "t3")
          assertEquals(metrics("min"), 6.7d, EPSILON)
          assertEquals(metrics("max"), 7d, EPSILON)
          assertEquals(metrics("normL1"), 7d, EPSILON)
          assertEquals(metrics("normL2"), 8d, EPSILON)
          assertEquals(metrics("numNonzeros"), 3d, EPSILON)
          assertEquals(metrics("mean"), 2.4d, EPSILON)
          assertEquals(metrics("variance"), 7d, EPSILON)

        case 4 =>
          assertEquals(feature, "f4" + Constants.DELIMITER)
          assertEquals(metrics("min"), 2.33d, EPSILON)
          assertEquals(metrics("max"), 4d, EPSILON)
          assertEquals(metrics("normL1"), 4d, EPSILON)
          assertEquals(metrics("normL2"), 5d, EPSILON)
          assertEquals(metrics("numNonzeros"), 89d, EPSILON)
          assertEquals(metrics("mean"), 3.6d, EPSILON)
          assertEquals(metrics("variance"), 0.5d, EPSILON)

        case _ => foundMatchedOne = false
      }

      if (foundMatchedOne) {
        count += 1
      }
    }

    assertEquals(count, 5)
  }
}

object ModelProcessingUtilsIntegTest {

  private val EPSILON = 1e-6

  val TASK_TYPE = TaskType.LOGISTIC_REGRESSION
  val GAME_OPTIMIZATION_CONFIGURATION: GameEstimator.GameOptimizationConfiguration = Map(
    ("fixed",
      FixedEffectOptimizationConfiguration(
        OptimizerConfig(OptimizerType.TRON, 10, 1e-1, constraintMap = None),
        NoRegularizationContext)),
    ("random1",
      RandomEffectOptimizationConfiguration(
        OptimizerConfig(OptimizerType.LBFGS, 20, 1e-2, constraintMap = None),
        L1RegularizationContext,
        regularizationWeight = 1D)),
    ("random2",
      RandomEffectOptimizationConfiguration(
        OptimizerConfig(OptimizerType.TRON, 30, 1e-3, constraintMap = None),
        L2RegularizationContext,
        regularizationWeight = 2D)))
}
