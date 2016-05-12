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
package com.linkedin.photon.ml.avro.model

import java.util.Random

import scala.collection.Map

import org.apache.hadoop.fs.Path
import org.testng.annotations.{DataProvider, Test}
import org.testng.Assert._

import com.linkedin.photon.ml.avro.Constants._
import com.linkedin.photon.ml.avro.data.NameAndTerm
import com.linkedin.photon.ml.constants.MathConst
import com.linkedin.photon.ml.model.{MatrixFactorizationModelTest, FixedEffectModel, Coefficients, RandomEffectModel}
import com.linkedin.photon.ml.test.{SparkTestUtils, TestTemplateWithTmpDir}

class ModelProcessingUtilsTest extends SparkTestUtils with TestTemplateWithTmpDir {

  @Test
  def testLoadAndSaveGameModels(): Unit = sparkTest("loadAndSaveRandomEffectModelToHDFS") {

    // Coefficients parameter
    val coefficientDimension = 1
    val coefficients = Coefficients.initializeZeroCoefficients(coefficientDimension)

    // Features
    val featureShardId = "featureShardId"
    val featureNameAndTermToIndexMap = Array.tabulate(coefficientDimension)( i => (NameAndTerm("n", "t"), i) ).toMap
    val featureShardIdToFeatureNameAndTermToIndexMapMap = Map(featureShardId -> featureNameAndTermToIndexMap)

    // Fixed effect model
    val coefficientsBC = sc.broadcast(coefficients)
    val fixedEffectModel = new FixedEffectModel(coefficientsBC, featureShardId)

    // Random effect model
    val numCoefficients = 5
    val randomEffectId = "randomEffectId"
    val coefficientsRDD = sc.parallelize(Seq.tabulate(numCoefficients)(i => (i.toString, coefficients)))
    val randomEffectModel = new RandomEffectModel(coefficientsRDD, randomEffectId, featureShardId)

    // GAME model
    val gameModel = Iterable(fixedEffectModel, randomEffectModel)

    // Default number of output files
    val numberOfOutputFilesForRandomEffectModel = 2
    val outputDir = getTmpDir
    val outputDirAsPath = new Path(outputDir)
    ModelProcessingUtils.saveGameModelsToHDFS(gameModel, featureShardIdToFeatureNameAndTermToIndexMapMap, outputDir,
      numberOfOutputFilesForRandomEffectModel, sc)

    val fs = outputDirAsPath.getFileSystem(sc.hadoopConfiguration)
    assertTrue(fs.exists(outputDirAsPath))

    // Check if the numberOfOutputFilesForRandomEffectModel parameter is working or not
    val randomEffectModelCoefficientsDir = new Path(outputDirAsPath,
      s"$RANDOM_EFFECT/$randomEffectId-$featureShardId/$COEFFICIENTS")
    val numRandomEffectModelFiles = fs.listStatus(randomEffectModelCoefficientsDir)
      .count(_.getPath.toString.contains("part"))
    assertTrue(numRandomEffectModelFiles == numberOfOutputFilesForRandomEffectModel,
      s"Expected number of random effect model files: $numberOfOutputFilesForRandomEffectModel, " +
          s"found: $numRandomEffectModelFiles")

    // Check if the models loaded correctly and they are the same as the models saved previously
    val loadedGameModel = ModelProcessingUtils.loadGameModelFromHDFS(featureShardIdToFeatureNameAndTermToIndexMapMap,
      outputDir, sc)
    loadedGameModel.foreach {
      case loadedFixedEffectModel: FixedEffectModel => assertEquals(loadedFixedEffectModel, fixedEffectModel)
      case loadedRandomEffectModel: RandomEffectModel => assertEquals(loadedRandomEffectModel, randomEffectModel)
    }
  }

  @DataProvider
  def matrixFactorizationConfigProvider():Array[Array[Any]] = {
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
  def testLoadAndSaveMatrixFactorizationModels(numLatentFactors: Int, numRows: Int, numCols: Int)
  : Unit = sparkTest("loadAndSaveRandomEffectModelToHDFS") {

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
