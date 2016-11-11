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

import scala.collection.Map
import scala.util.Random

import breeze.linalg.DenseVector
import org.apache.hadoop.fs.Path
import org.testng.Assert._
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.avro.Constants._
import com.linkedin.photon.ml.avro.data.NameAndTerm
import com.linkedin.photon.ml.avro.model.ModelProcessingUtils
import com.linkedin.photon.ml.constants.MathConst
import com.linkedin.photon.ml.model._
import com.linkedin.photon.ml.supervised.classification.LogisticRegressionModel
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import com.linkedin.photon.ml.test.{SparkTestUtils, TestTemplateWithTmpDir}
import com.linkedin.photon.ml.util._

class ModelProcessingUtilsTest extends SparkTestUtils with TestTemplateWithTmpDir {

  /**
   * Test that we can load a simple Game model with fixed and random effects, given a feature index
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
    def extractNamedFeatures(glm: GeneralizedLinearModel, featureIndexLoader: IndexMapLoader) = {
      val featureIndex = featureIndexLoader.indexMapForRDD()
      glm.coefficients.means match {
        case (vector: breeze.linalg.DenseVector[Double]) =>
          vector.iterator.map { case (index, value) => (value, featureIndex.getFeatureName(index)) }
        case (vector: breeze.linalg.SparseVector[Double]) =>
          // activeIterator to iterate over the non-zeros
          vector.activeIterator.map { case (index, value) => (value, featureIndex.getFeatureName(index)) }
      }
    }

    (gameModel.toMap zip loadedGameModel.toMap) foreach {

      case ((n1: String, m1: FixedEffectModel), (n2: String, m2: FixedEffectModel)) =>

        assertEquals(n1, n2)
        val fixedMap1 = extractNamedFeatures(m1.model, featureIndexLoaders(n1))
        val fixedMap2 = extractNamedFeatures(m2.model, newFeatureIndexLoaders(n2))
        assert(fixedMap1 sameElements fixedMap2)

      case ((n1: String, m1: RandomEffectModel), (n2: String, m2: RandomEffectModel)) =>

        assertEquals(n1, n2)
        m1.modelsRDD.join(m2.modelsRDD).mapValues{ case (glm1, glm2) =>
          val fixedMap1 = extractNamedFeatures(glm1, featureIndexLoaders(n1))
          val fixedMap2 = extractNamedFeatures(glm2, newFeatureIndexLoaders(n2))
          assert(fixedMap1 sameElements fixedMap2)
        }.collect

      case _ =>
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
