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

import java.util.Random

import breeze.linalg.DenseVector
import org.apache.spark.sql.DataFrame
import org.mockito.Mockito._
import org.testng.Assert._
import org.testng.annotations.{BeforeMethod, DataProvider, Test}

import com.linkedin.photon.ml.evaluation.Evaluator
import com.linkedin.photon.ml.evaluation.Evaluator.EvaluationResults
import com.linkedin.photon.ml.model.GameModel
import com.linkedin.photon.ml.normalization.NormalizationContext
import com.linkedin.photon.ml.optimization.game._
import com.linkedin.photon.ml.test.Assertions.assertIterableEqualsWithTolerance

/**
 * Unit tests for GameEstimatorEvaluationFunction
 */
class GameEstimatorEvaluationFunctionTest {

  val gameModel = mock(classOf[GameModel])
  val evaluationResults = mock(classOf[EvaluationResults])
  val evaluator = mock(classOf[Evaluator])
  val estimator = mock(classOf[GameEstimator])
  val trainingData = mock(classOf[DataFrame])
  val validationData = mock(classOf[DataFrame])
  val optimizationConfiguration = mock(classOf[GameModelOptimizationConfiguration])
  val evaluationFunction = new GameEstimatorEvaluationFunction(
    estimator, optimizationConfiguration, trainingData, validationData)
  val tol = 1e-7
  val random = new Random(1)
  val eval = random.nextDouble
  val regWeights = Array.fill[Double](8) { random.nextDouble }

  @Test
  def testVectorToConfiguration(): Unit = {
    val configuration = GameModelOptimizationConfiguration(
      Map("a" -> GLMOptimizationConfiguration(regularizationWeight = regWeights(0))),
      Map("b" -> GLMOptimizationConfiguration(regularizationWeight = regWeights(1))),
      Map("c" -> FactoredRandomEffectOptimizationConfiguration(
        GLMOptimizationConfiguration(regularizationWeight = regWeights(2)),
        GLMOptimizationConfiguration(regularizationWeight = regWeights(3)),
        MFOptimizationConfiguration(1, 1))))

    val evaluationFunction = new GameEstimatorEvaluationFunction(estimator, configuration, trainingData, validationData)

    val hypers = DenseVector(regWeights(4), regWeights(5), regWeights(6), regWeights(7))
    val newConfiguration = evaluationFunction.vectorToConfiguration(hypers)

    assertEquals(newConfiguration.fixedEffectOptimizationConfiguration("a").regularizationWeight, regWeights(4))
    assertEquals(newConfiguration.randomEffectOptimizationConfiguration("b").regularizationWeight, regWeights(5))
    assertEquals(newConfiguration.factoredRandomEffectOptimizationConfiguration("c")
      .randomEffectOptimizationConfiguration.regularizationWeight, regWeights(6))
    assertEquals(newConfiguration.factoredRandomEffectOptimizationConfiguration("c")
      .latentFactorOptimizationConfiguration.regularizationWeight, regWeights(7))
  }

  @DataProvider
  def configurationProvider = {
    Array(
      Array(GameModelOptimizationConfiguration(
        Map("a" -> GLMOptimizationConfiguration(regularizationWeight = regWeights(0))),
        Map(),
        Map()),
        DenseVector(regWeights(0))),

      Array(GameModelOptimizationConfiguration(
        Map("a" -> GLMOptimizationConfiguration(regularizationWeight = regWeights(0))),
        Map("b" -> GLMOptimizationConfiguration(regularizationWeight = regWeights(1))),
        Map()),
        DenseVector(regWeights(0), regWeights(1))),

      Array(GameModelOptimizationConfiguration(
        Map("a" -> GLMOptimizationConfiguration(regularizationWeight = regWeights(0))),
        Map("b" -> GLMOptimizationConfiguration(regularizationWeight = regWeights(1))),
        Map("c" -> FactoredRandomEffectOptimizationConfiguration(
          GLMOptimizationConfiguration(regularizationWeight = regWeights(2)),
          GLMOptimizationConfiguration(regularizationWeight = regWeights(3)),
          MFOptimizationConfiguration(1, 1)))),
        DenseVector(regWeights(0), regWeights(1), regWeights(2), regWeights(3)))
    )
  }

  @Test(dataProvider = "configurationProvider")
  def testVectorizeParams(configuration: GameModelOptimizationConfiguration, expected: DenseVector[Double]): Unit = {
    val gameResult = (gameModel, Some(evaluationResults), configuration)
    val result = evaluationFunction.vectorizeParams(gameResult)

    assertEquals(result.length, expected.length)
    assertIterableEqualsWithTolerance(result.toArray, expected.toArray, tol)
  }
}
