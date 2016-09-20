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
package com.linkedin.photon.ml.model

import java.util.Random

import breeze.linalg.Vector
import org.apache.spark.SparkContext
import org.testng.Assert._
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.data.{KeyValueScore, GameDatum}
import com.linkedin.photon.ml.constants.MathConst
import com.linkedin.photon.ml.test.SparkTestUtils


/**
 * test the matrix factorization model
 */
class MatrixFactorizationModelTest extends SparkTestUtils {

  import MatrixFactorizationModelTest._

  @DataProvider
  def matrixFactorizationConfigProvider():Array[Array[Any]] = {
    Array(
      Array(1, 1, 0),
      Array(1, 0, 1),
      Array(1, 1, 1),
      Array(5, 10, 10)
    )
  }

  @Test(dataProvider = "matrixFactorizationConfigProvider")
  def testScore(numLatentFactors: Int, numRows: Int, numCols: Int)
  : Unit = sparkTest("testScoreForMatrixFactorizationModel") {

    // Meta data, the actual value doesn't matter for this test
    val rowEffectType = "rowEffectType"
    val colEffectType = "colEffectType"

    // The row and column latent factor generator
    val random = new Random(MathConst.RANDOM_SEED)
    def randomRowLatentFactorGenerator = generateRandomLatentFactor(numLatentFactors, random)
    def randomColLatentFactorGenerator = generateRandomLatentFactor(numLatentFactors, random)

    val rowLatentFactors = Array.tabulate(numRows)(i => (i.toString, randomRowLatentFactorGenerator))
    val colLatentFactors = Array.tabulate(numCols)(j => (j.toString, randomColLatentFactorGenerator))
    val rowRange = 0 until numRows
    val colRange = 0 until numCols

    // generate the synthetic game data and scores
    val (gameData, syntheticScores) = rowRange.zip(colRange).map { case (row, col) =>
      val rowId = row.toString
      val colId = col.toString
      val randomEffectIdToIndividualIdMap = Map(rowEffectType -> rowId, colEffectType -> colId)
      val gameDatum = new GameDatum(response = 1.0, offsetOpt = None, weightOpt = None, featureShardContainer = Map(),
        idTypeToValueMap = randomEffectIdToIndividualIdMap)
      val score = rowLatentFactors(row)._2.dot(colLatentFactors(col)._2)
      (gameDatum, score)
    }
      .zipWithIndex
      .map { case ((gameDatum, score), uniqueId) => ((uniqueId.toLong, gameDatum), (uniqueId.toLong, score)) }
      .unzip

    // Construct the matrix
    val randomMFModel = new MatrixFactorizationModel(rowEffectType, colEffectType,
      rowLatentFactors = sc.parallelize(rowLatentFactors), colLatentFactors = sc.parallelize(colLatentFactors))

    val expectedScores = new KeyValueScore(sc.parallelize(syntheticScores))
    val computedScores = randomMFModel.score(sc.parallelize(gameData))

    assertEquals(computedScores, expectedScores)
  }

  @Test(dataProvider = "matrixFactorizationConfigProvider")
  def testEquals(numLatentFactors: Int, numRows: Int, numCols: Int)
  : Unit = sparkTest("testEqualsForMatrixFactorizationModel") {

    // A random matrix factorization model
    val rowEffectType = "rowEffectType"
    val colEffectType = "colEffectType"
    val random = new Random(MathConst.RANDOM_SEED)
    def randomRowLatentFactorGenerator = generateRandomLatentFactor(numLatentFactors, random)
    def randomColLatentFactorGenerator = generateRandomLatentFactor(numLatentFactors, random)
    val randomMFModel = generateMatrixFactorizationModel(numRows, numCols, rowEffectType, colEffectType,
      randomRowLatentFactorGenerator, randomColLatentFactorGenerator, sc)

    // Should equal to itself
    assertEquals(randomMFModel, randomMFModel)

    // Should equal to the matrix factorization model with same meta data and latent factors
    val randomMFModelCopy = new MatrixFactorizationModel(randomMFModel.rowEffectType, randomMFModel.colEffectType,
      randomMFModel.rowLatentFactors, randomMFModel.colLatentFactors)
    assertEquals(randomMFModel, randomMFModelCopy)

    // Should not equal to the matrix factorization model with different row effect Id
    val rowEffectType1 = "rowEffectType1"
    val randomMFModelWithDiffRowEffectId = new MatrixFactorizationModel(rowEffectType1, randomMFModel.colEffectType,
      randomMFModel.rowLatentFactors, randomMFModel.colLatentFactors)
    assertNotEquals(randomMFModel, randomMFModelWithDiffRowEffectId)

    // Should not equal to the matrix factorization model with different col effect Id
    val colEffectType1 = "colEffectType1"
    val randomMFModelWithDiffColEffectId = new MatrixFactorizationModel(randomMFModel.rowEffectType, colEffectType1,
      randomMFModel.rowLatentFactors, randomMFModel.colLatentFactors)
    assertNotEquals(randomMFModel, randomMFModelWithDiffColEffectId)

    // Should not equal to the matrix factorization model with different latent factors
    val zeroMFModel = generateMatrixFactorizationModel(numRows, numCols, rowEffectType, colEffectType,
      generateZerosLatentFactor(numLatentFactors), generateZerosLatentFactor(numLatentFactors), sc)
    assertNotEquals(randomMFModel, zeroMFModel)
  }
}

object MatrixFactorizationModelTest {
  // Generate a latent factor of zeros
  protected[ml] def generateZerosLatentFactor(numLatentFactors: Int): Vector[Double] = {
    Vector.zeros[Double](numLatentFactors)
  }

  // Generate a latent factor with random numbers
  protected[ml] def generateRandomLatentFactor(numLatentFactors: Int, random: Random): Vector[Double] = {
    Vector.fill(numLatentFactors)(random.nextDouble())
  }

  // Generate a matrix factorization model with the given specs
  protected[ml] def generateMatrixFactorizationModel(
    numRows: Int,
    numCols: Int,
    rowEffectType: String,
    colEffectType: String,
    rowFactorGenerator: => Vector[Double],
    colFactorGenerator: => Vector[Double],
    sparkContext: SparkContext): MatrixFactorizationModel = {

    val rowLatentFactors =
      sparkContext.parallelize(Seq.tabulate(numRows)(i => (i.toString, rowFactorGenerator)))
    val colLatentFactors =
      sparkContext.parallelize(Seq.tabulate(numCols)(j => (j.toString, colFactorGenerator)))
    new MatrixFactorizationModel(rowEffectType, colEffectType, rowLatentFactors, colLatentFactors)
  }
}
