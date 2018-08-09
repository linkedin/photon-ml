/*
 * Copyright 2018 LinkedIn Corp. All rights reserved.
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
package com.linkedin.photon.ml.hyperparameter.criteria

import breeze.linalg.DenseVector
import org.apache.spark.rdd.RDD
import org.mockito.Mockito.mock
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.evaluation.{Evaluator, EvaluatorType}
import com.linkedin.photon.ml.test.Assertions.assertIterableEqualsWithTolerance

/**
 * Test cases for the ConfidenceBound class
 */
class ConfidenceBoundTest {

  import ConfidenceBoundTest._

  val tol = 1e-3

  /**
   * Test data
   */
  @DataProvider
  def modelDataProvider() =
    Array(
      Array(DenseVector(1.0, 2.0, 3.0), DenseVector(1.0, 2.0, 3.0), 0),
      Array(DenseVector(-4.0, 5.0, -6.0), DenseVector(3.0, 2.0, 1.0), 1))

  /**
   * Unit tests for [[ConfidenceBound.apply]]
   */
  @Test(dataProvider = "modelDataProvider")
  def testApply(mu: DenseVector[Double], sigma: DenseVector[Double], testSetIndex: Int): Unit = {

    val upperConfidenceBound = new ConfidenceBound(EVALUATOR_AUC)
    val predictedMax = upperConfidenceBound(mu, sigma)

    val expectedMax = testSetIndex match {
      case 0 => DenseVector(3.0000, 4.8284, 6.4641)
      case 1 => DenseVector(-0.5359, 7.8284, -4.0000)
    }

    assertIterableEqualsWithTolerance(predictedMax.toArray, expectedMax.toArray, tol)

    val lowerConfidenceBound = new ConfidenceBound(EVALUATOR_RMSE)
    val predictedMin = lowerConfidenceBound(mu, sigma)

    val expectedMin = testSetIndex match {
      case 0 => DenseVector(-1.0000, -0.8284, -0.4641)
      case 1 => DenseVector(-7.4641, 2.1716, -8.0000)
    }

    assertIterableEqualsWithTolerance(predictedMin.toArray, expectedMin.toArray, tol)
  }
}

object ConfidenceBoundTest {
  val EVALUATOR_AUC: Evaluator = new Evaluator {

    override protected[ml] val labelAndOffsetAndWeights: RDD[(Long, (Double, Double, Double))] =
      mock(classOf[RDD[(Long, (Double, Double, Double))]])
    override val evaluatorType: EvaluatorType = EvaluatorType.AUC
    override protected[ml] def evaluateWithScoresAndLabelsAndWeights(
        scoresAndLabelsAndWeights: RDD[(Long, (Double, Double, Double))]): Double = 0.0

    override def betterThan(score1: Double, score2: Double): Boolean = score1 > score2
  }

  val EVALUATOR_RMSE: Evaluator = new Evaluator {

    override protected[ml] val labelAndOffsetAndWeights: RDD[(Long, (Double, Double, Double))] =
      mock(classOf[RDD[(Long, (Double, Double, Double))]])
    override val evaluatorType: EvaluatorType = EvaluatorType.RMSE
    override protected[ml] def evaluateWithScoresAndLabelsAndWeights(
        scoresAndLabelsAndWeights: RDD[(Long, (Double, Double, Double))]): Double = 0.0

    override def betterThan(score1: Double, score2: Double): Boolean = score1 < score2
  }
}
