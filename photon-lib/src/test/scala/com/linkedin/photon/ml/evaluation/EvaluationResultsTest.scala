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
package com.linkedin.photon.ml.evaluation

import org.apache.spark.rdd.RDD
import org.testng.Assert._
import org.testng.annotations.Test

import com.linkedin.photon.ml.constants.MathConst

/**
 * Unit tests for [[EvaluationResults]].
 */
class EvaluationResultsTest {

  /**
   * Test that invalid input will cause an error during [[EvaluationResults]] construction.
   */
  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testInvalidInput(): Unit = {

    val primaryEvaluator: EvaluatorType = EvaluatorType.AUC
    val evaluations: Map[EvaluatorType, (Double, Option[RDD[(String, Double)]])] = Map()

    EvaluationResults(evaluations, primaryEvaluator)
  }

  /**
   * Test that the primary evaluation metric can be correctly fetched from some [[EvaluationResults]].
   */
  @Test
  def testPrimaryEvaluation(): Unit = {

    val evaluation: Double = 1.23
    val primaryEvaluator: EvaluatorType = EvaluatorType.AUC
    val evaluations: Map[EvaluatorType, (Double, Option[RDD[(String, Double)]])] =
      Map(primaryEvaluator -> (evaluation, None))
    val evaluationResults: EvaluationResults = EvaluationResults(evaluations, primaryEvaluator)

    assertEquals(evaluation, evaluationResults.primaryEvaluation, MathConst.EPSILON)
  }
}
