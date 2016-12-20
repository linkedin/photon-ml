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
package com.linkedin.photon.ml.avro

import scala.util.Random

import org.apache.hadoop.fs.Path
import org.testng.Assert._
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.avro.Constants._
import com.linkedin.photon.ml.avro.model.ModelProcessingUtils
import com.linkedin.photon.ml.constants.MathConst
import com.linkedin.photon.ml.model._
import com.linkedin.photon.ml.supervised.classification.LogisticRegressionModel
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import com.linkedin.photon.ml.test.{SparkTestUtils, TestTemplateWithTmpDir}
import com.linkedin.photon.ml.util._

/**
 * Unit tests for model processing utilities.
 */
class ModelProcessingUtilsTest extends SparkTestUtils with TestTemplateWithTmpDir {
  /**
   * Test that we can load a simple Game model with fixed and random effects, given a feature index.
   */
  @Test
  def testLoadAndSaveGameModels(): Unit = sparkTest("loadAndSaveRandomEffectModelToHDFS") {

    // Features: we have two feature spaces, one for the fix model, and one for the random model
    // Each model has its own, separate feature space, but feature values can be shared between spaces.
    // Features shared between spaces have a unique name, but possibly different indices.
    val numFeaturesPerModel = Map("fixed" -> 10, "random" -> 15)
    val featureIndexLoaders = numFeaturesPerModel.mapValues { numFeatures =>
      DefaultIndexMapLoader(sc, (0 until numFeatures).map(i => Utils.getFeatureKey("n" + i, "t")))
    }

    // Fixed effect model
    val fixedEffectCoefficients = Coefficients(numFeaturesPerModel("fixed"))(1,2,5)(1,2,5)
    val glm: GeneralizedLinearModel = new LogisticRegressionModel(fixedEffectCoefficients)
    val fixedEffectModel = new FixedEffectModel(sc.broadcast(glm), "fixed")

    // Random effect model
    val randomEffectCoefficients = Coefficients(numFeaturesPerModel("random"))(1,5,9)(1,5,9)
    val glmRE: GeneralizedLinearModel = new LogisticRegressionModel(randomEffectCoefficients)
    val glmReRDD = sc.parallelize((0 until 2).map(i => (i.toString, glmRE)))
    val randomEffectModel = new RandomEffectModel(glmReRDD, "randomEffectType", "random")

    // GAME model
    val gameModel = new GAMEModel(Map("fixed" -> fixedEffectModel, "random" -> randomEffectModel))

    // Default number of output files
    val numberOfOutputFilesForRandomEffectModel = 2
    val outputDir = getTmpDir
    val outputDirAsPath = new Path(outputDir)

    ModelProcessingUtils.saveGameModelsToHDFS(
      gameModel,
      featureIndexLoaders,
      outputDir,
      numberOfOutputFilesForRandomEffectModel,
      sc)

    val fs = outputDirAsPath.getFileSystem(sc.hadoopConfiguration)
    assertTrue(fs.exists(outputDirAsPath))

    // Check if the numberOfOutputFilesForRandomEffectModel parameter is working or not
    val randomEffectModelCoefficientsDir = new Path(outputDirAsPath, s"$RANDOM_EFFECT/random/$COEFFICIENTS")
    val numRandomEffectModelFiles = fs.listStatus(randomEffectModelCoefficientsDir)
      .count(_.getPath.toString.contains("part"))
    assertEquals(numRandomEffectModelFiles, numberOfOutputFilesForRandomEffectModel,
      s"Expected number of random effect model files: $numberOfOutputFilesForRandomEffectModel, " +
        s"found: $numRandomEffectModelFiles")

    // Check if the models loaded correctly and they are the same as the models saved previously
    // The first value returned is the feature index, which we don't need here
    val (loadedGameModel, _) = ModelProcessingUtils.loadGameModelFromHDFS(Some(featureIndexLoaders), outputDir, sc)
    assertEquals(gameModel, loadedGameModel)
  }

  /**
   * Test that we can load a model even if we don't have a feature index. In that case, the feature index is
   * generated as part of loading the model, and it will not directly match the feature index before the save.
   */
  @Test
  def testLoadGameModelsWithoutFeatureIndex(): Unit = sparkTest("loadAndSaveRandomEffectModelToHDFS2") {

    // Features: we have two feature spaces, one for the fix model, and one for the random model
    // Each model has its own, separate feature space, but feature values can be shared between spaces.
    // Features shared between spaces have a unique name, but possibly different indices.
    val numFeaturesPerModel = Map("fixed" -> 10, "random" -> 15)
    val featureIndexLoaders = numFeaturesPerModel.mapValues { numFeatures =>
      DefaultIndexMapLoader(sc, (0 until numFeatures).map(i => Utils.getFeatureKey("n" + i, "t")))
    }.map(identity) // .map(identity) needed because of: https://issues.scala-lang.org/browse/SI-7005

    // Fixed effect model
    val fixedEffectCoefficients = Coefficients(numFeaturesPerModel("fixed"))(1,2,5)(11,21,51)
    val glm: GeneralizedLinearModel = new LogisticRegressionModel(fixedEffectCoefficients)
    val fixedEffectModel = new FixedEffectModel(sc.broadcast(glm), "fixed")

    // Random effect model
    val randomEffectCoefficients = Coefficients(numFeaturesPerModel("random"))(1,5,7)(101,501,901)
    val glmRE: GeneralizedLinearModel = new LogisticRegressionModel(randomEffectCoefficients)
    val glmReRDD = sc.parallelize((0 until 2).map(i => (i.toString, glmRE)))
    val randomEffectModel = new RandomEffectModel(glmReRDD, "randomEffectType", "random")

    // GAME model
    val modelDir = getTmpDir
    val gameModel = new GAMEModel(Map("fixed" -> fixedEffectModel, "random" -> randomEffectModel))

    ModelProcessingUtils.saveGameModelsToHDFS(gameModel, featureIndexLoaders, modelDir, 2, sc)

    // Check if the models loaded correctly and they are the same as the models saved previously
    // The first value returned is the feature index, which we don't need here
    val (loadedGameModel, newFeatureIndexLoaders) = ModelProcessingUtils.loadGameModelFromHDFS(None, modelDir, sc)

    // Since the feature index after the load is not the same as before the save, we need to calculate an
    // invariant that survives the save/load. That invariant is Map[feature value, feature name].
    def features(glm: GeneralizedLinearModel, featureIndexLoader: IndexMapLoader): Array[(String, Double)] =
      ModelProcessingUtils.extractGLMFeatures(glm, featureIndexLoader.indexMapForRDD())

    (gameModel.toMap zip loadedGameModel.toMap) foreach {

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
   * Test that we can extract all features from a Game model correctly.
   */
  @Test
  def testExtractGameModelFeatures(): Unit = sparkTest("extractGameModelFeatures") {

    // Features: we have two feature spaces, one for the fix model, and one for the random model
    // Each model has its own, separate feature space, but feature values can be shared between spaces.
    // Features shared between spaces have a unique name, but possibly different indices.
    val numFeaturesPerModel = Map("fixedFeatures" -> 10, "RE1Features" -> 10, "RE2Features" -> 10)
    val featureNames = numFeaturesPerModel.mapValues { numFeatures =>
      (0 until numFeatures).map(i => Utils.getFeatureKey("n" + i, "t")) }
    val featureIndexLoaders = featureNames.map { case (modelType, modelFeatures) =>
      (modelType, DefaultIndexMapLoader(sc, modelFeatures))
    }.map(identity) // .map(identity) needed because of: https://issues.scala-lang.org/browse/SI-7005

    // Fixed effect model
    val glm = new LogisticRegressionModel(Coefficients(numFeaturesPerModel("fixedFeatures"))(1,2,5)(11,21,51))
    val fixedEffectModel = new FixedEffectModel(sc.broadcast(glm), "fixedFeatures")

    // Random effect 1 has 2 items
    val numFeaturesRE1 = numFeaturesPerModel("RE1Features")
    val RE1Item1 = Coefficients(numFeaturesRE1)(1,5,7)(111,511,911)
    val glmRE11: GeneralizedLinearModel = new LogisticRegressionModel(RE1Item1)
    val RE1Item2 = Coefficients(numFeaturesRE1)(1,2)(112,512)
    val glmRE12: GeneralizedLinearModel = new LogisticRegressionModel(RE1Item2)

    val glmRE1RDD = sc.parallelize(List(("RE1Item1", glmRE11), ("RE1Item2", glmRE12)))
    val RE1Model = new RandomEffectModel(glmRE1RDD, "randomEffectModel1", "RE1Features")

    // Random effect 2 has 3 items (of a different kind)
    val numFeaturesRE2 = numFeaturesPerModel("RE2Features")
    val RE2Item1 = Coefficients(numFeaturesRE2)(3,4,6)(321,421,621)
    val glmRE21: GeneralizedLinearModel = new LogisticRegressionModel(RE2Item1)
    val RE2Item2 = Coefficients(numFeaturesRE2)(4,5)(322,422)
    val glmRE22: GeneralizedLinearModel = new LogisticRegressionModel(RE2Item2)
    val RE2Item3 = Coefficients(numFeaturesRE2)(2,7,8)(323,423,523)
    val glmRE23: GeneralizedLinearModel = new LogisticRegressionModel(RE2Item3)

    val glmRE2RDD = sc.parallelize(List(("RE2Item1", glmRE21), ("RE2Item2", glmRE22), ("RE2Item3", glmRE23)))
    val RE2Model = new RandomEffectModel(glmRE2RDD, "randomEffectModel2", "RE2Features")

    // This Game model has 1 fixed effect, and 2 different random effect models
    val gameModel = new GAMEModel(Map("fixed" -> fixedEffectModel, "RE1" -> RE1Model, "RE2" -> RE2Model))

    // Structure of files for this model on HDFS is:
    //    hdfs://hostname:port/tmp/GameLaserModelTest/gameModel/fixed-effect/fixed/coefficients/part-00000.avro
    //    hdfs://hostname:port/tmp/GameLaserModelTest/gameModel/fixed-effect/fixed/id-info
    //    hdfs://hostname:port/tmp/GameLaserModelTest/gameModel/random-effect/RE1/coefficients/_SUCCESS
    //    hdfs://hostname:port/tmp/GameLaserModelTest/gameModel/random-effect/RE1/coefficients/part-00000.avro
    //    hdfs://hostname:port/tmp/GameLaserModelTest/gameModel/random-effect/RE1/coefficients/part-00001.avro
    //    hdfs://hostname:port/tmp/GameLaserModelTest/gameModel/random-effect/RE1/id-info
    //    hdfs://hostname:port/tmp/GameLaserModelTest/gameModel/random-effect/RE2/coefficients/_SUCCESS
    //    hdfs://hostname:port/tmp/GameLaserModelTest/gameModel/random-effect/RE2/coefficients/part-00000.avro
    //    hdfs://hostname:port/tmp/GameLaserModelTest/gameModel/random-effect/RE2/coefficients/part-00001.avro
    //    hdfs://hostname:port/tmp/GameLaserModelTest/gameModel/random-effect/RE2/id-info

    val modelDir = getTmpDir
    ModelProcessingUtils.saveGameModelsToHDFS(gameModel, featureIndexLoaders, modelDir, 2, sc)

    // Check if the models loaded correctly and they are the same as the models saved previously
    // The first value returned is the feature index, which we don't need here
    val (loadedGameModel, newFeatureIndexLoaders) = ModelProcessingUtils.loadGameModelFromHDFS(None, modelDir, sc)

    // Let's extract all features from the Game model...
    val features = ModelProcessingUtils.extractGameModelFeatures(sc, loadedGameModel, newFeatureIndexLoaders)

    // ... and verify the models
    features.foreach {
      case ((FIXED_EFFECT, "fixed"), modelRDD) =>
        val calculated: Array[(String, Double)] = modelRDD.collect()(0)._2
        val ans = List(1, 2, 5).map(i => featureNames("fixedFeatures")(i)) zip List(11,21,51)
        assert(calculated sameElements ans)

      case ((RANDOM_EFFECT, "RE1"), modelRDD) =>
        val features = featureNames("RE1Features")
        modelRDD.collect.foreach {

          case ("RE1Item1", coefficients) =>
            assert(coefficients sameElements (List(1, 5, 7).map(i => features(i)) zip List(111, 511, 911)))

          case ("RE1Item2", coefficients) =>
            assert(coefficients sameElements (List(1, 2).map(i => features(i)) zip List(112, 512)))
        }

      case ((RANDOM_EFFECT, "RE2"), modelRDD) =>
        val features = featureNames("RE2Features")
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
    sparkTest("loadAndSaveRandomEffectModelToHDFS") {
      import MatrixFactorizationModelTest._

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
}
