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

import scala.math.floor

import breeze.linalg.DenseVector
import org.testng.Assert._
import org.testng.annotations.Test

import com.linkedin.photon.ml.hyperparameter.EvaluationFunction
import com.linkedin.photon.ml.hyperparameter.estimators.kernels.Matern52
import com.linkedin.photon.ml.util.DoubleRange

/**
 * Unit tests for [[RandomSearch]].
 */
class RandomSearchTest {

  import RandomSearchTest._

  /**
   * Test that [[RandomSearch]] can generate multiple points in the search space.
   */
  @Test
  def testFind(): Unit = {

    val searcher = new RandomSearch[TestModel](RANGES, EVALUATION_FUNCTION, DISCRETE_PARAMS, KERNEL, SEED)
    val candidates = searcher.find(N)

    assertEquals(candidates.length, N)
    assertEquals(candidates.toSet.size, N)
    assertTrue(candidates.forall(_.params.toArray.forall(x => x >= LOWER && x <= UPPER)))
    assertTrue(candidates.forall(c => c.params(0) == floor(c.params(0))))
  }

  /**
   * Test that prior observations don't affect [[RandomSearch]].
   */
  @Test(dependsOnMethods = Array[String]("testFind"))
  def testFindWithPriors(): Unit = {

    val searcher = new RandomSearch[TestModel](RANGES, EVALUATION_FUNCTION, DISCRETE_PARAMS, KERNEL, SEED)
    val priorSearcher = new RandomSearch[TestModel](RANGES, EVALUATION_FUNCTION, DISCRETE_PARAMS, KERNEL, SEED)
    val observation1 = (DenseVector(1.0, 1.0, 1.0), 0.1)
    val observation2 = (DenseVector(2.0, 2.0, 2.0), 0.2)
    val observation3 = (DenseVector(3.0, 3.0, 3.0), 0.3)
    val observations = Seq(observation1, observation3)
    val priorObservations = Seq(observation2)

    val candidates = searcher.find(N)
    val priorsCandidates = priorSearcher.findWithPriors(N, observations, priorObservations)

    assertEquals(priorsCandidates.length, N)
    assertEquals(priorsCandidates.toSet.size, N)

    candidates.zip(priorsCandidates).foreach { case (candidate, priorCandidate) =>
      assertTrue(candidate.params == priorCandidate.params)
      assertEquals(candidate.evaluation, priorCandidate.evaluation, TOLERANCE)
    }
  }
}

object RandomSearchTest {

  val SEED = 1L
  val DIM = 10
  val N = 25
  val LOWER = 1e-5
  val UPPER = 1e5
  val RANGES: Seq[DoubleRange] = Seq.fill(DIM)(DoubleRange(LOWER, UPPER))
  val DISCRETE_PARAMS = Seq(0)
  val KERNEL = new Matern52
  val TOLERANCE = 1E-12

  case class TestModel(params: DenseVector[Double], evaluation: Double)

  val EVALUATION_FUNCTION = new EvaluationFunction[TestModel] {

    def apply(hyperParameters: DenseVector[Double]): (Double, TestModel) = {
      (0.0, TestModel(hyperParameters, 0.0))
    }

    def vectorizeParams(result: TestModel): DenseVector[Double] = result.params
    def getEvaluationValue(result: TestModel): Double = result.evaluation
  }
}
