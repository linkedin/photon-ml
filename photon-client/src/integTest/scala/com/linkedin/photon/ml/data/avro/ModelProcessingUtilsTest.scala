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

import scala.collection.immutable.IndexedSeq
import scala.util.Random

import breeze.linalg.Vector
import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.testng.Assert._
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.constants.{MathConst, StorageLevel}
import com.linkedin.photon.ml.estimators.GameParams
import com.linkedin.photon.ml.model._
import com.linkedin.photon.ml.optimization.game.GLMOptimizationConfiguration
import com.linkedin.photon.ml.supervised.classification.LogisticRegressionModel
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import com.linkedin.photon.ml.test.{SparkTestUtils, TestTemplateWithTmpDir}
import com.linkedin.photon.ml.util._

/**
 * Unit tests for model processing utilities.
 */
class ModelProcessingUtilsTest extends SparkTestUtils with TestTemplateWithTmpDir {

  /**
   * Ancillary function to setup some params, as required by the unit tests.
   *
   * @param updates A sequence of pairs (Params field name, field value) to customize the Params returned
   * @return An instance of Params
   */
  def setupParams(updates: (String, Any)*): GameParams = {

    import GLMOptimizationConfiguration.{SPLITTER => S}

    val params = new GameParams

    // Some default optimization configurations
    val feConfig1 = GLMOptimizationConfiguration(s"10${S}1e-2${S}1.0${S}0.3${S}TRON${S}L2")
    val reConfig1 = GLMOptimizationConfiguration(s"20${S}1e-2${S}1.0${S}0.3${S}LBFGS${S}L1")
    val reConfig2 = GLMOptimizationConfiguration(s"30${S}1e-2${S}1.0${S}0.2${S}TRON${S}L2")

    params.fixedEffectOptimizationConfigurations = Array(Map("fixed" -> feConfig1))
    params.randomEffectOptimizationConfigurations = Array(Map("random1" -> reConfig1, "random2" -> reConfig2))

    // Now update the created Params with the updates specified in the call
    updates.foreach {
      case (name: String, value: Any) =>
        params.getClass.getMethods.find(_.getName == name + "_$eq").get.invoke(params, value.asInstanceOf[AnyRef])
    }

    params
  }

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
   * @return (GAMEModel, featureIndexLoaders, featureNames)
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

    val (gameModel, featureIndexLoaders, _) = makeGameModel()

    // Default number of output files
    val numberOfOutputFilesForRandomEffectModel = 2
    val params = setupParams(("numberOfOutputFilesForRandomEffectModel", numberOfOutputFilesForRandomEffectModel))
    val outputDir = getTmpDir
    val outputDirAsPath = new Path(outputDir)

    ModelProcessingUtils.saveGameModelsToHDFS(
      gameModel,
      featureIndexLoaders,
      outputDir,
      params,
      sc)

    val fs = outputDirAsPath.getFileSystem(sc.hadoopConfiguration)
    assertTrue(fs.exists(outputDirAsPath))

    // Check if the numberOfOutputFilesForRandomEffectModel parameter is working or not
    val randomEffectModelCoefficientsDir =
      new Path(outputDirAsPath, s"${AvroConstants.RANDOM_EFFECT}/RE1/${AvroConstants.COEFFICIENTS}")
    val numRandomEffectModelFiles = fs.listStatus(randomEffectModelCoefficientsDir)
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
   * Test that we can load a model even if we don't have a feature index. In that case, the feature index is
   * generated as part of loading the model, and it will not directly match the feature index before the save.
   */
  @Test
  def testLoadGameModelsWithoutFeatureIndex(): Unit = sparkTest("testLoadGameModelsWithoutFeatureIndex") {

    val (params, modelDir) = (setupParams(), getTmpDir)
    val (gameModel, featureIndexLoaders, _) = makeGameModel()

    ModelProcessingUtils.saveGameModelsToHDFS(gameModel, featureIndexLoaders, modelDir, params, sc)

    val (loadedGameModel, newFeatureIndexLoaders) = ModelProcessingUtils.loadGameModelFromHDFS(
      sc,
      modelDir,
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

    val (params, modelDir) = (setupParams(), getTmpDir)
    val (gameModel, featureIndexLoaders, featureNames) = makeGameModel()

    ModelProcessingUtils.saveGameModelsToHDFS(gameModel, featureIndexLoaders, modelDir, params, sc)

    // Check if the models loaded correctly and they are the same as the models saved previously
    // The first value returned is the feature index, which we don't need here
    val (loadedGameModel, newFeatureIndexLoaders) = ModelProcessingUtils.loadGameModelFromHDFS(
      sc,
      modelDir,
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

  @Test(dataProvider = "matrixFactorizationConfigProvider")
  def testLoadAndSaveMatrixFactorizationModels(numLatentFactors: Int, numRows: Int, numCols: Int): Unit =
    sparkTest("testLoadAndSaveMatrixFactorizationModels") {

      // Generate a latent factor with random numbers
      def generateRandomLatentFactor(numLatentFactors: Int, random: Random): Vector[Double] =
        Vector.fill(numLatentFactors)(random.nextDouble())

      // Generate a matrix factorization model with the given specs
      def generateMatrixFactorizationModel(
          numRows: Int,
          numCols: Int,
          rowEffectType: String,
          colEffectType: String,
          rowFactorGenerator: => Vector[Double],
          colFactorGenerator: => Vector[Double],
          sc: SparkContext): MatrixFactorizationModel = {

        val rowLatentFactors =
          sc.parallelize(Seq.tabulate(numRows)(i => (i.toString, rowFactorGenerator)))
        val colLatentFactors =
          sc.parallelize(Seq.tabulate(numCols)(j => (j.toString, colFactorGenerator)))
        new MatrixFactorizationModel(rowEffectType, colEffectType, rowLatentFactors, colLatentFactors)
      }

      // Meta data
      val rowEffectType = "rowEffectType"
      val colEffectType = "colEffectType"

      // Generate the random matrix
      val random = new Random(MathConst.RANDOM_SEED)
      def randomRowLatentFactorGenerator = generateRandomLatentFactor(numLatentFactors, random)
      def randomColLatentFactorGenerator = generateRandomLatentFactor(numLatentFactors, random)
      val randomMFModel = generateMatrixFactorizationModel(numRows, numCols, rowEffectType, colEffectType,
        randomRowLatentFactorGenerator, randomColLatentFactorGenerator, sc)

      val tmpDir = getTmpDir
      val numOutputFiles = 1
      // Save the model to HDFS
      ModelProcessingUtils.saveMatrixFactorizationModelToHDFS(randomMFModel, tmpDir, numOutputFiles, sc)
      // Load the model from HDFS
      val loadedRandomMFModel =
        ModelProcessingUtils.loadMatrixFactorizationModelFromHDFS(tmpDir, rowEffectType, colEffectType, sc)
      assertEquals(loadedRandomMFModel, randomMFModel)
    }

  /**
   * Test that we can save and load model metadata.
   *
   * TODO: this is incomplete - need to check that more parameters are loaded back correctly
   */
  @Test
  def testSaveAndLoadGameModelMetadata(): Unit = sparkTest("testSaveAndLoadGameModelMetadata") {

    val params = setupParams()
    ModelProcessingUtils.saveGameModelMetadataToHDFS(sc, params, "/tmp")
    val params2 = ModelProcessingUtils.loadGameModelMetadataFromHDFS(sc, "/tmp")
    assertEquals(params.taskType, params2.taskType)
  }
}
