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

import breeze.linalg.DenseVector
import org.testng.Assert._
import org.testng.annotations.Test

import com.linkedin.photon.ml.hyperparameter.EvaluationFunction
import com.linkedin.photon.ml.util.DoubleRange

/**
 * Test cases for the RandomSearch class
 */
class RandomSearchTest {

  val seed = 1L
  val dim = 10
  val n = 25
  val lower = 1e-5
  val upper = 1e5
  val ranges: Seq[DoubleRange] = Seq.fill(dim)(DoubleRange(lower, upper))

  case class TestModel(params: DenseVector[Double], evaluation: Double)

  val evaluationFunction = new EvaluationFunction[TestModel] {

    def apply(hyperParameters: DenseVector[Double]): (Double, TestModel) = {
      (0.0, TestModel(hyperParameters, 0.0))
    }

    def vectorizeParams(result: TestModel): DenseVector[Double] = result.params
    def getEvaluationValue(result: TestModel): Double = result.evaluation
  }

  val searcher = new RandomSearch[TestModel](ranges, evaluationFunction, seed)

  @Test
  def testFind(): Unit = {

    val candidates = searcher.find(n)

    assertEquals(candidates.length, n)
    assertTrue(candidates.forall(_.params.toArray.forall(x => x >= lower && x <= upper)))
    assertEquals(candidates.toSet.size, n)
  }
}
