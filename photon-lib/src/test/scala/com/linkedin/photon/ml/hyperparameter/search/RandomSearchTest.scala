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
package com.linkedin.photon.ml.hyperparameter.search

import breeze.linalg.{DenseVector, norm}
import org.testng.Assert._
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.hyperparameter.EvaluationFunction
import com.linkedin.photon.ml.hyperparameter.estimators.kernels.Matern52

/**
 * Unit tests for [[RandomSearch]].
 */
class RandomSearchTest {

  import RandomSearchTest._

  val searcher = new RandomSearch[TestModel](DIM, EVALUATION_FUNCTION, DISCRETE_PARAMS, KERNEL, SEED)

  @DataProvider
  def priorDataProvider: Array[Array[Any]] = {

    val candidate1 = (DenseVector(0.25, 0.125, 0.999), 0.1)
    val candidate2 = (DenseVector(0.2, 0.2, 0.2), 0.2)
    val candidate3 = (DenseVector(0.3, 0.3, 0.3), -0.3)
    val candidate4 = (DenseVector(0.2, 0.2, 0.2), 0.4)
    val candidate5 = (DenseVector(0.3, 0.3, 0.3), -0.4)
    val observations = Seq(candidate1, candidate2, candidate3)
    val priorObservations = Seq(candidate4, candidate5)

    Array(Array(observations, priorObservations))
  }

  /**
   * Test that [[RandomSearch]] can generate multiple points in the search space.
   */
  @Test(dataProvider = "priorDataProvider")
  def testFindWithPriors(
      observations: Seq[(DenseVector[Double], Double)],
      priorObservations: Seq[(DenseVector[Double], Double)]): Unit = {

    val candidates = searcher.findWithPriors(N, observations, priorObservations)

    assertEquals(candidates.length, N)
    assertEquals(candidates.toSet.size, N)
    assertTrue(candidates.forall(_.params.toArray.forall(x => x >= 0 && x < 1)))
  }

  /**
   * Test that [[RandomSearch]] can generate multiple points in the search space.
   */
  @Test(dataProvider = "priorDataProvider", dependsOnMethods = Array[String]("testFindWithPriors"))
  def testFindWithPriorObservations(
      observations: Seq[(DenseVector[Double], Double)],
      priorObservations: Seq[(DenseVector[Double], Double)]): Unit = {

    val candidates = searcher.findWithPriorObservations(N, priorObservations)

    assertEquals(candidates.length, N)
    assertEquals(candidates.toSet.size, N)
    assertTrue(candidates.forall(_.params.toArray.forall(x => x >= 0 && x < 1)))
  }

  /**
   * Test that observations and prior observations don't affect [[RandomSearch]].
   */
  @Test(dataProvider = "priorDataProvider", dependsOnMethods = Array[String]("testFindWithPriorObservations"))
  def testFind(
      observations: Seq[(DenseVector[Double], Double)],
      priorObservations: Seq[(DenseVector[Double], Double)]): Unit = {

    val searcher1 = new RandomSearch[TestModel](DIM, EVALUATION_FUNCTION, DISCRETE_PARAMS, KERNEL, SEED)
    val candidates1 = searcher1.find(N)

    assertEquals(candidates1.length, N)
    assertEquals(candidates1.toSet.size, N)
    assertTrue(candidates1.forall(_.params.toArray.forall(x => x >= 0 && x < 1)))

    val searcher2 = new RandomSearch[TestModel](DIM, EVALUATION_FUNCTION, DISCRETE_PARAMS, KERNEL, SEED)
    val candidates2 = searcher2.findWithPriors(N, observations, priorObservations)

    candidates1.zip(candidates2).foreach { case (candidate1, candidate2) =>
      assertTrue(candidate1.params == candidate2.params)
      assertEquals(candidate1.evaluation, candidate2.evaluation, TOLERANCE)
    }
  }

  /**
   * Test that the candidates of integer hyper-parameters are discretized correctly.
   */
  @Test(dataProvider = "priorDataProvider")
  def testDiscretizeCandidate(
      observations: Seq[(DenseVector[Double], Double)],
      priorObservations: Seq[(DenseVector[Double], Double)]): Unit = {

    val candidate = observations.head._1
    val expectedData = DenseVector(0.2, 0.125, 8.0 / 9.0)

    val candidateWithDiscrete = searcher.discretizeCandidate(candidate, DISCRETE_PARAMS)
    assertEquals(norm(candidateWithDiscrete), norm(expectedData), TOLERANCE)
  }
}

object RandomSearchTest {

  val SEED = 1L
  val DIM = 3
  val N = 5
  val DISCRETE_PARAMS = Map(0 -> 5, 2 -> 9)
  val KERNEL = new Matern52
  val TOLERANCE = 1E-12

  case class TestModel(params: DenseVector[Double], evaluation: Double)

  val EVALUATION_FUNCTION: EvaluationFunction[TestModel] = new EvaluationFunction[TestModel] {

    def apply(hyperParameters: DenseVector[Double]): (Double, TestModel) = {
      (0.0, TestModel(hyperParameters, 0.0))
    }

    def convertObservations(results: Seq[TestModel]): Seq[(DenseVector[Double], Double)] = {
      results.map { result =>
        val candidate = vectorizeParams(result)
        val value = getEvaluationValue(result)
        (candidate, value)
      }
    }
    def vectorizeParams(result: TestModel): DenseVector[Double] = result.params
    def getEvaluationValue(result: TestModel): Double = result.evaluation
  }
}
