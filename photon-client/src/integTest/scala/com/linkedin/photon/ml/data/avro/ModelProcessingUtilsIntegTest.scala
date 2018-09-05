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

import breeze.linalg.{DenseVector, SparseVector}
import org.apache.avro.file.DataFileReader
import org.apache.avro.specific.SpecificDatumReader
import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.apache.spark.storage.StorageLevel
import org.testng.Assert._
import org.testng.annotations.Test

import com.linkedin.photon.avro.generated.FeatureSummarizationResultAvro
import com.linkedin.photon.ml.Types.{CoordinateId, FeatureShardId, REId}
import com.linkedin.photon.ml.cli.game.training.GameTrainingDriver
import com.linkedin.photon.ml.estimators.GameEstimator
import com.linkedin.photon.ml.index.{DefaultIndexMap, DefaultIndexMapLoader, IndexMap, IndexMapLoader}
import com.linkedin.photon.ml.model._
import com.linkedin.photon.ml.optimization._
import com.linkedin.photon.ml.optimization.game.{FixedEffectOptimizationConfiguration, RandomEffectOptimizationConfiguration}
import com.linkedin.photon.ml.stat.BasicStatisticalSummary
import com.linkedin.photon.ml.supervised.classification.LogisticRegressionModel
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import com.linkedin.photon.ml.test.{SparkTestUtils, TestTemplateWithTmpDir}
import com.linkedin.photon.ml.util._
import com.linkedin.photon.ml.{Constants, TaskType}

/**
 * Integration tests for [[ModelProcessingUtils]].
 */
class ModelProcessingUtilsIntegTest extends SparkTestUtils with TestTemplateWithTmpDir {

  import ModelProcessingUtilsIntegTest._

  /**
   * Test that we can load a simple GAME model with fixed and random effects.
   */
  @Test
  def testLoadAndSaveGameModels(): Unit = sparkTest("testLoadAndSaveGameModels") {

    val (gameModel, featureIndexLoaders) = makeGameModel(sc)
    val outputDir = new Path(getTmpDir)

    // Save the model to HDFS
    ModelProcessingUtils.saveGameModelToHDFS(
      sc,
      outputDir,
      gameModel,
      TaskType.LOGISTIC_REGRESSION,
      GAME_OPTIMIZATION_CONFIGURATION,
      randomEffectModelFileLimit = None,
      featureIndexLoaders,
      VectorUtils.DEFAULT_SPARSITY_THRESHOLD)

    // Load the model from HDFS
    val loadedGameModel = ModelProcessingUtils.loadGameModelFromHDFS(
      sc,
      outputDir,
      StorageLevel.DISK_ONLY,
      featureIndexLoaders)

    // Check that the model loaded correctly and that it is identical to the model saved
    assertTrue(gameModel == loadedGameModel)
  }
  import ModelProcessingUtilsIntegTest._

  /**
   * Test that we can load a subset of the GAME model coordinates.
   */
  @Test
  def testLoadPartialModel(): Unit = sparkTest("testLoadPartialModel") {

    val numCoordinatesToLoad = 2
    val (gameModel, featureIndexLoaders) = makeGameModel(sc)
    val outputDir = new Path(getTmpDir)

    // Save the model to HDFS
    ModelProcessingUtils.saveGameModelToHDFS(
      sc,
      outputDir,
      gameModel,
      TaskType.LOGISTIC_REGRESSION,
      GAME_OPTIMIZATION_CONFIGURATION,
      randomEffectModelFileLimit = None,
      featureIndexLoaders,
      VectorUtils.DEFAULT_SPARSITY_THRESHOLD)

    // Load the model from HDFS, but ignore the second random effect model
    val loadedGameModelMap = ModelProcessingUtils
      .loadGameModelFromHDFS(
        sc,
        outputDir,
        StorageLevel.DISK_ONLY,
        featureIndexLoaders,
        Some(SHARD_NAMES.take(numCoordinatesToLoad).toSet))
      .toMap

    // Check that only some of the coordinates were loaded
    assertEquals(loadedGameModelMap.size, numCoordinatesToLoad)
    for (i <- 0 until numCoordinatesToLoad) {
      assertTrue(loadedGameModelMap.contains(SHARD_NAMES(i)))
    }
    for (i <- numCoordinatesToLoad until SHARD_NAMES.length) {
      assertFalse(loadedGameModelMap.contains(SHARD_NAMES(i)))
    }
  }

  /**
   * Test that we can save a GAME model with custom sparsity threshold.
   */
  @Test
  def testSparsityThreshold(): Unit = sparkTest("testSparsityThreshold") {

    // Model sparsity threshold
    val modelSparsityThreshold = FIXED_COEFFICIENTS.means.valuesIterator.drop(2).next() + 1

    val (gameModel, featureIndexLoaders) = makeGameModel(sc)
    val outputDir = new Path(getTmpDir)

    // Save the model to HDFS
    ModelProcessingUtils.saveGameModelToHDFS(
      sc,
      outputDir,
      gameModel,
      TaskType.LOGISTIC_REGRESSION,
      GAME_OPTIMIZATION_CONFIGURATION,
      randomEffectModelFileLimit = None,
      featureIndexLoaders,
      modelSparsityThreshold)

    // Load the model from HDFS
    val loadedGameModel = ModelProcessingUtils.loadGameModelFromHDFS(
      sc,
      outputDir,
      StorageLevel.DISK_ONLY,
      featureIndexLoaders)

    // Check that some of the values have been filtered out by the new threshold for non-zero values
    loadedGameModel.getModel("fixed") match {
      case Some(model: FixedEffectModel) =>
        assertEquals(
          model.modelBroadcast.value.coefficients.means.valuesIterator.toSet - 0D,
          FIXED_COEFFICIENTS.means.valuesIterator.filter(_ > modelSparsityThreshold).toSet)

      case other =>
        fail(s"Unexpected model: $other")
    }
  }

  /**
   * Test that we can save a GAME model to a limited number of files on HDFS.
   */
  @Test
  def testRandomEffectModelFilesLimit(): Unit = sparkTest("testRandomEffectModelFilesLimit") {

    // Default number of output files
    val numberOfOutputFilesForRandomEffectModel = 2

    val (gameModel, featureIndexLoaders) = makeGameModel(sc)
    val outputDir = new Path(getTmpDir)

    // Save the model to HDFS
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

    val randomEffect1ModelCoefficientsDir = new Path(
      outputDir,
      s"${AvroConstants.RANDOM_EFFECT}/RE1/${AvroConstants.COEFFICIENTS}")
    val randomEffect2ModelCoefficientsDir = new Path(
      outputDir,
      s"${AvroConstants.RANDOM_EFFECT}/RE2/${AvroConstants.COEFFICIENTS}")
    val numRandomEffect1ModelFiles = fs
      .listStatus(randomEffect1ModelCoefficientsDir)
      .count(_.getPath.toString.contains("part"))
    val numRandomEffect2ModelFiles = fs
      .listStatus(randomEffect2ModelCoefficientsDir)
      .count(_.getPath.toString.contains("part"))

    // Test that the number of output files for the random effect models has been limited
    assertEquals(
      numRandomEffect1ModelFiles,
      numberOfOutputFilesForRandomEffectModel,
      s"Mismatch in number of random effect model files: expected $numberOfOutputFilesForRandomEffectModel " +
        s"but found: $numRandomEffect1ModelFiles")
    assertEquals(
      numRandomEffect2ModelFiles,
      numberOfOutputFilesForRandomEffectModel,
      s"Mismatch in number of random effect model files: expected $numberOfOutputFilesForRandomEffectModel " +
        s"but found: $numRandomEffect2ModelFiles")
  }

  /**
   * Test that if a model has features not present in index maps, they're ignored when loading.
   */
  @Test
  def testFeaturesMissingFromIndexMap(): Unit = sparkTest("testFeaturesMissingFromIndexMap") {

    val (gameModel, indexMapLoaders) = makeGameModel(sc)
    val outputDir = new Path(getTmpDir)

    // Remove a feature from each index map
    val modifiedIndexMapLoaders = indexMapLoaders.mapValues { indexMapLoader =>
      val featureNameToIdMap = indexMapLoader.indexMapForDriver().asInstanceOf[DefaultIndexMap].featureNameToIdMap

      new DefaultIndexMapLoader(sc, featureNameToIdMap - getFeatureName(1))
    }

    // Save the model to HDFS using the original index maps
    ModelProcessingUtils.saveGameModelToHDFS(
      sc,
      outputDir,
      gameModel,
      TaskType.LOGISTIC_REGRESSION,
      GAME_OPTIMIZATION_CONFIGURATION,
      randomEffectModelFileLimit = None,
      indexMapLoaders,
      VectorUtils.DEFAULT_SPARSITY_THRESHOLD)

    // Load the model from HDFS using the modified index maps
    val loadedGameModel = ModelProcessingUtils.loadGameModelFromHDFS(
      sc,
      outputDir,
      StorageLevel.DISK_ONLY,
      modifiedIndexMapLoaders)

    // Extract features from the GAME model
    val features = extractGameModelFeatures(loadedGameModel, modifiedIndexMapLoaders)

    // Verify that the removed feature is no longer present in the models
    features.foreach {

      case (FIXED_SHARD_NAME, featuresMap) =>
        val calculated = featuresMap.head._2

        assertTrue(calculated.sameElements(extractCoefficients(FIXED_COEFFICIENTS, toDrop = 2)))

      case (RE1_SHARD_NAME, featuresMap) =>
        featuresMap.foreach {

          case ("RE1Item1", coefficients) =>
            assertTrue(coefficients.sameElements(extractCoefficients(RE11_COEFFICIENTS, toDrop = 1)))

          case ("RE1Item2", coefficients) =>
            assertTrue(coefficients.sameElements(extractCoefficients(RE12_COEFFICIENTS, toDrop = 1)))
        }

      case (RE2_SHARD_NAME, featuresMap) =>
        featuresMap.foreach {

          case ("RE2Item1", coefficients) =>
            assertTrue(coefficients.sameElements(extractCoefficients(RE21_COEFFICIENTS, toDrop = 1)))

          case ("RE2Item2", coefficients) =>
            assertTrue(coefficients.sameElements(extractCoefficients(RE22_COEFFICIENTS, toDrop = 1)))

          case ("RE2Item3", coefficients) =>
            assertTrue(coefficients.sameElements(extractCoefficients(RE23_COEFFICIENTS, toDrop = 1)))
        }
    }
  }

  /**
   * Test that if the index maps have features not present in the model, they're 0 when loaded.
   */
  @Test
  def testExtraFeaturesInIndexMap(): Unit = sparkTest("testExtraFeaturesInIndexMap") {

    val (gameModel, indexMapLoaders) = makeGameModel(sc)
    val outputDir = new Path(getTmpDir)

    // Add a new feature to each index map
    val modifiedIndexMapLoaders = indexMapLoaders.mapValues { indexMapLoader =>
      val featureNameToIdMap = indexMapLoader.indexMapForDriver().asInstanceOf[DefaultIndexMap].featureNameToIdMap

      new DefaultIndexMapLoader(sc, featureNameToIdMap + ((getFeatureName(NUM_FEATURES + 1), NUM_FEATURES + 1)))
    }

    // Save the model to HDFS using the original index maps
    ModelProcessingUtils.saveGameModelToHDFS(
      sc,
      outputDir,
      gameModel,
      TaskType.LOGISTIC_REGRESSION,
      GAME_OPTIMIZATION_CONFIGURATION,
      randomEffectModelFileLimit = None,
      indexMapLoaders,
      VectorUtils.DEFAULT_SPARSITY_THRESHOLD)

    // Load the model from HDFS using the modified index maps
    val loadedGameModel = ModelProcessingUtils.loadGameModelFromHDFS(
      sc,
      outputDir,
      StorageLevel.DISK_ONLY,
      modifiedIndexMapLoaders)

    // Extract features from the GAME model
    val features = extractGameModelFeatures(loadedGameModel, modifiedIndexMapLoaders)

    // Verify that the extra feature is not present in any of the models
    features.foreach {

      case (FIXED_SHARD_NAME, featuresMap) =>
        val calculated = featuresMap.head._2

        assertTrue(calculated.sameElements(extractCoefficients(FIXED_COEFFICIENTS, toDrop = 1)))

      case (RE1_SHARD_NAME, featuresMap) =>
        featuresMap.foreach {

          case ("RE1Item1", coefficients) =>
            assertTrue(coefficients.sameElements(extractCoefficients(RE11_COEFFICIENTS)))

          case ("RE1Item2", coefficients) =>
            assertTrue(coefficients.sameElements(extractCoefficients(RE12_COEFFICIENTS)))
        }

      case (RE2_SHARD_NAME, featuresMap) =>
        featuresMap.foreach {

          case ("RE2Item1", coefficients) =>
            assertTrue(coefficients.sameElements(extractCoefficients(RE21_COEFFICIENTS)))

          case ("RE2Item2", coefficients) =>
            assertTrue(coefficients.sameElements(extractCoefficients(RE22_COEFFICIENTS)))

          case ("RE2Item3", coefficients) =>
            assertTrue(coefficients.sameElements(extractCoefficients(RE23_COEFFICIENTS)))
        }
    }
  }

  /**
   * Test that we can save and load model metadata.
   */
  @Test
  def testSaveAndLoadGameModelMetadata(): Unit = sparkTest("testSaveAndLoadGameModelMetadata") {

    val outputDir = new Path(getTmpDir)

    // Save model metadata
    ModelProcessingUtils.saveGameModelMetadataToHDFS(sc, outputDir, TASK_TYPE, GAME_OPTIMIZATION_CONFIGURATION)

    // TODO: This test is incomplete - need to check that all parameters are loaded correctly.
    assertEquals(
      TASK_TYPE,
      ModelProcessingUtils.loadGameModelMetadataFromHDFS(sc, outputDir)(GameTrainingDriver.trainingTask))
  }

  /**
   * Test computing and writing out [[BasicStatisticalSummary]].
   */
  @Test
  def testWriteBasicStatistics(): Unit = sparkTest("testWriteBasicStatistics") {

    val dim: Int = 6
    val interceptIndex: Int = dim - 1
    val minVector = VectorUtils.toSparseVector(Array((0, 1.5), (3, 6.7), (4, 2.33), (5, 1D)), dim)
    val maxVector = VectorUtils.toSparseVector(Array((0, 10D), (3, 7D), (4, 4D), (5, 1D)), dim)
    val normL1Vector = VectorUtils.toSparseVector(Array((0, 1D), (3, 7D), (4, 4D), (5, 10D)), dim)
    val normL2Vector = VectorUtils.toSparseVector(Array((0, 2D), (3, 8D), (4, 5D), (5, 10D)), dim)
    val numNonzeros = VectorUtils.toSparseVector(Array((0, 6D), (3, 3D), (4, 89D), (5, 100D)), dim)
    val meanVector = VectorUtils.toSparseVector(Array((0, 1.1), (3, 2.4), (4, 3.6), (5, 1D)), dim)
    val varianceVector = VectorUtils.toSparseVector(Array((0, 1D), (3, 7D), (4, 0.5), (5, 0D)), dim)

    val summary = BasicStatisticalSummary(
      meanVector,
      varianceVector,
      count = 100L,
      numNonzeros,
      maxVector,
      minVector,
      normL1Vector,
      normL2Vector,
      meanVector,
      Some(interceptIndex))

    val indexMap: IndexMap = new DefaultIndexMap(
      Map(
        Utils.getFeatureKey("f0", "") -> 0,
        Utils.getFeatureKey("f1", "t1") -> 1,
        Utils.getFeatureKey("f2", "") -> 2,
        Utils.getFeatureKey("f3", "t3") -> 3,
        Utils.getFeatureKey("f4", "") -> 4,
        Constants.INTERCEPT_KEY -> 5))

    val outputDir = new Path(getTmpDir, "summary-output")
    ModelProcessingUtils.writeBasicStatistics(sc, summary, outputDir, indexMap)

    val reader = DataFileReader.openReader[FeatureSummarizationResultAvro](
      new File(outputDir.toString + "/part-00000.avro"),
      new SpecificDatumReader[FeatureSummarizationResultAvro]())

    val count = Iterator
      .continually {
        val record = reader.next()
        val featureKey = Utils.getFeatureKey(record.getFeatureName, record.getFeatureTerm)
        val featureIndex = indexMap(featureKey)
        val metrics = record.getMetrics.map {case (key, value) => (String.valueOf(key), value)}

        assertNotEquals(featureIndex, interceptIndex)
        assertEquals(featureKey, indexMap.getFeatureName(featureIndex).get)
        assertEquals(metrics("min"), minVector(featureIndex), EPSILON)
        assertEquals(metrics("max"), maxVector(featureIndex), EPSILON)
        assertEquals(metrics("normL1"), normL1Vector(featureIndex), EPSILON)
        assertEquals(metrics("normL2"), normL2Vector(featureIndex), EPSILON)
        assertEquals(metrics("numNonzeros"), numNonzeros(featureIndex), EPSILON)
        assertEquals(metrics("mean"), meanVector(featureIndex), EPSILON)
        assertEquals(metrics("variance"), varianceVector(featureIndex), EPSILON)

        featureIndex
      }
      .takeWhile(_ => reader.hasNext)
      .length

    // Add one to count, since the value of reader is always evaluated once before hasNext is checked. However, also
    // subtract one from count, since intercept should be skipped.
    assertEquals(count + 1, dim - 1)
  }
}

object ModelProcessingUtilsIntegTest {

  private val FIXED_SHARD_NAME = "fixed"
  private val RE1_SHARD_NAME = "RE1"
  private val RE2_SHARD_NAME = "RE2"
  private val SHARD_NAMES = Seq(FIXED_SHARD_NAME, RE1_SHARD_NAME, RE2_SHARD_NAME)
  private val GAME_OPTIMIZATION_CONFIGURATION: GameEstimator.GameOptimizationConfiguration = Map(
    (FIXED_SHARD_NAME,
      FixedEffectOptimizationConfiguration(
        OptimizerConfig(OptimizerType.TRON, 10, 1e-1, constraintMap = None),
        NoRegularizationContext)),
    (RE1_SHARD_NAME,
      RandomEffectOptimizationConfiguration(
        OptimizerConfig(OptimizerType.LBFGS, 20, 1e-2, constraintMap = None),
        L1RegularizationContext,
        regularizationWeight = 1D)),
    (RE2_SHARD_NAME,
      RandomEffectOptimizationConfiguration(
        OptimizerConfig(OptimizerType.TRON, 30, 1e-3, constraintMap = None),
        L2RegularizationContext,
        regularizationWeight = 2D)))

  private val NUM_FEATURES = 7
  private val FEATURE_NAMES = (0 until NUM_FEATURES).map(getFeatureName)

  private val FIXED_COEFFICIENTS = CoefficientsTest.denseCoefficients(0D, 11D, 21D, 31D, 41D, 51D, 61D)
  private val RE11_COEFFICIENTS = CoefficientsTest.sparseCoefficients(NUM_FEATURES)(1, 2)(111D, 211D)
  private val RE12_COEFFICIENTS = CoefficientsTest.sparseCoefficients(NUM_FEATURES)(1, 3)(112D, 312D)
  private val RE21_COEFFICIENTS = CoefficientsTest.sparseCoefficients(NUM_FEATURES)(1, 4)(121D, 421D)
  private val RE22_COEFFICIENTS = CoefficientsTest.sparseCoefficients(NUM_FEATURES)(1, 5)(122D, 522D)
  private val RE23_COEFFICIENTS = CoefficientsTest.sparseCoefficients(NUM_FEATURES)(1, 6)(123D, 623D)

  private val EPSILON = 1e-6
  private val TASK_TYPE = TaskType.LOGISTIC_REGRESSION

  /**
   * Generate a toy GAME model for subsequent tests. This GAME model trains a logistic regression problem. It has one
   * fixed effect and two random effect coordinates.
   *
   * @note Each coordinate uses its own feature space
   * @note We give each coordinate and its feature shard the same name because it makes it easier to test
   *
   * @param sc The [[SparkContext]] for the test
   * @return A tuple of (toy GAME model, index maps for model)
   */
  def makeGameModel(sc: SparkContext): (GameModel, Map[FeatureShardId, IndexMapLoader]) = {

    // Build index maps
    val featureIndexLoaders = SHARD_NAMES.map((_, DefaultIndexMapLoader(sc, FEATURE_NAMES))).toMap

    // Fixed effect
    val fixedEffectModel = new FixedEffectModel(sc.broadcast(LogisticRegressionModel(FIXED_COEFFICIENTS)), "fixed")

    // First random effect
    val glmRE1RDD = sc.parallelize(
      List(
        ("RE1Item1", LogisticRegressionModel(RE11_COEFFICIENTS)),
        ("RE1Item2", LogisticRegressionModel(RE12_COEFFICIENTS))))
    val RE1Model = new RandomEffectModel(glmRE1RDD, "randomEffectModel1", "RE1")

    // Second random effect
    val glmRE2RDD = sc.parallelize(
      List(
        ("RE2Item1", LogisticRegressionModel(RE21_COEFFICIENTS)),
        ("RE2Item2", LogisticRegressionModel(RE22_COEFFICIENTS)),
        ("RE2Item3", LogisticRegressionModel(RE23_COEFFICIENTS))))
    val RE2Model = new RandomEffectModel(glmRE2RDD, "randomEffectModel2", "RE2")

    val model = GameModel(SHARD_NAMES.zip(Seq(fixedEffectModel, RE1Model, RE2Model)): _*)

    (model, featureIndexLoaders)
  }

  /**
   * Generate a test feature name based on a given index.
   *
   * @param i Some index
   * @return A feature name
   */
  def getFeatureName(i: Int): String = Utils.getFeatureKey("n" + i.toString, "t")

  /**
   * Extract (feature key, feature value) pairs for all non-zero coefficient means in a [[Coefficients]] object.
   * Optionally drop some of the coefficients.
   *
   * @param coefficients The [[Coefficients]]
   * @param toDrop The number of coefficients to drop, if any
   * @return A [[Seq]] of (feature key, feature value) pairs
   */
  def extractCoefficients(coefficients: Coefficients, toDrop: Int = 0): Seq[(String, Double)] =
    coefficients
      .means
      .activeIterator
      .drop(toDrop)
      .toSeq.map { case (index, value) =>
        (getFeatureName(index), value)
      }

  /**
   * Extract (feature key, feature value) pairs for all non-zero feature coefficients in each GLM in a GAME model.
   *
   * @param gameModel The GAME model from which to extract feature data
   * @param featureIndexLoaders Map of [[IndexMapLoader]] objects to use for loading feature space index maps for each
   *                            coordinate
   * @return A [[Map]] of coordinate ID to [[Map]] of entity ID to extracted (feature name, feature value) pairs (fixed
   *         effect models will have only one entry in the map, and the entity ID will match the coordinate ID)
   */
  def extractGameModelFeatures(
      gameModel: GameModel,
      featureIndexLoaders: Map[FeatureShardId, IndexMapLoader]): Map[CoordinateId, Map[REId, Array[(String, Double)]]] =
    gameModel
      .toMap
      .map {
        case (fixedEffect: String, model: FixedEffectModel) =>
          val featureIndex = featureIndexLoaders(model.featureShardId).indexMapForDriver()

          (fixedEffect, Map((fixedEffect, extractGLMFeatures(model.model, featureIndex))))

        case (randomEffect: String, model: RandomEffectModel) =>
          // Each random effect has a feature space, referred to by a shard id
          val featureShardId = model.featureShardId
          val featureIndexLoader = featureIndexLoaders(featureShardId)
          val featuresMapRDD = model.modelsRDD.mapPartitions { iter =>
            // Calling mapPartitions allows us to only need to serialize this map once per executor
            val featureIndexes = featureIndexLoader.indexMapForRDD()

            iter.map { case (rEId, glm) =>
              (rEId, extractGLMFeatures(glm, featureIndexes))
            }
          }

        (randomEffect, featuresMapRDD.collect().toMap)

        case (modelType, _) =>
          throw new RuntimeException(s"Unknown model type: $modelType")
      }

  /**
   * Extract (feature key, feature value) pairs for all non-zero feature coefficients in a GLM.
   *
   * @param glm The GLM from which to extract (feature key, feature value) pairs
   * @param featureIndex The index map for the feature space
   * @return An array of (feature key, feature value) pairs for all active (non-zero) features in the GLM
   */
  def extractGLMFeatures(glm: GeneralizedLinearModel, featureIndex: IndexMap): Array[(String, Double)] = {

    val coefficients: Iterator[(Int, Double)] = glm.coefficients.means match {
      case vector: DenseVector[Double] => vector.iterator
      case vector: SparseVector[Double] => vector.activeIterator
    }

    // Get (feature name, feature value) pairs for all non-zero coefficients of the GLM (flatMap filters out None values
    // that can result if a feature is missing from the index map)
    coefficients
      .flatMap { case (index, value) => featureIndex.getFeatureName(index).map((_, value)) }
      .filter { case (_, value) => !MathUtils.isAlmostZero(value) }
      .toArray
  }
}
