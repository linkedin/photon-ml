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
import org.mockito.Mockito._
import org.testng.Assert._
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.Types.UniqueSampleId

/**
 * Unit tests for [[EvaluationSuite]].
 */
class EvaluationSuiteTest {

  import EvaluationSuiteTest._

  @DataProvider
  def invalidInput: Array[Array[Any]] = {

    val emptyEvaluators: Set[Evaluator] = Set()
    val goodEvaluators: Set[Evaluator] = Set(MOCK_AUC_EVALUATOR)

    val primaryEvaluator: Evaluator = MOCK_AUPR_EVALUATOR

    Array(
      Array(emptyEvaluators, primaryEvaluator, MOCK_RDD),
      Array(goodEvaluators, primaryEvaluator, MOCK_RDD))
  }

  /**
   * Test that invalid input will cause an error during [[EvaluationSuite]] construction.
   *
   * @param evaluators The [[Set]] of [[Evaluator]] objects
   * @param primaryEvaluator The 'primary' [[Evaluator]] (e.g. the one which should be used for model selection)
   * @param labelAndOffsetAndWeights The labels, offsets, and weights of the validation data which should be used to
   *                                 evaluate metrics
   */
  @Test(dataProvider = "invalidInput", expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testCheckInvariants(
      evaluators: Set[Evaluator],
      primaryEvaluator: Evaluator,
      labelAndOffsetAndWeights: RDD[(UniqueSampleId, (Double, Double, Double))]): Unit =
    new EvaluationSuite(evaluators, primaryEvaluator, labelAndOffsetAndWeights)

  @DataProvider
  def invalidApplyInput: Array[Array[Any]] = {

    val duplicateEvaluators: Seq[Evaluator] = Seq(MOCK_AUC_EVALUATOR, MOCK_AUC_EVALUATOR)

    Array(Array(duplicateEvaluators, MOCK_RDD))
  }

  /**
   * Test that invalid input will cause an error during [[EvaluationSuite]] construction.
   *
   * @param evaluators The [[Set]] of [[Evaluator]] objects
   * @param labelAndOffsetAndWeights The labels, offsets, and weights of the validation data which should be used to
   *                                 evaluate metrics
   */
  @Test(dataProvider = "invalidApplyInput", expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testInvalidApply(
      evaluators: Seq[Evaluator],
      labelAndOffsetAndWeights: RDD[(UniqueSampleId, (Double, Double, Double))]): Unit =
    EvaluationSuite(evaluators, labelAndOffsetAndWeights)

  /**
   * Test that an [[EvaluationSuite]] can be properly constructed using the apply helper function.
   */
  @Test
  def testValidApply(): Unit = {

    val evaluators: Seq[Evaluator] = Seq(MOCK_AUC_EVALUATOR, MOCK_AUPR_EVALUATOR)
    val evaluationSuite: EvaluationSuite = EvaluationSuite(evaluators, MOCK_RDD)

    assertEquals(evaluationSuite.primaryEvaluator, evaluators.head)
  }
}

object EvaluationSuiteTest {

  private val MOCK_RDD = mock(classOf[RDD[(UniqueSampleId, (Double, Double, Double))]])
  private val MOCK_AUC_EVALUATOR = new MockEvaluator(EvaluatorType.AUC)
  private val MOCK_AUPR_EVALUATOR = new MockEvaluator(EvaluatorType.AUPR)

  class MockEvaluator(override val evaluatorType: EvaluatorType) extends Evaluator {

    type ScoredData = Double

    override def evaluate(scoresAndLabelsAndWeights: RDD[Double]): Double =
      scoresAndLabelsAndWeights.sum()
  }
}
