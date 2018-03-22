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

import breeze.linalg.{DenseMatrix, DenseVector}
import org.apache.spark.rdd.RDD
import org.mockito.Mockito._
import org.testng.Assert._
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.evaluation.{Evaluator, EvaluatorType}
import com.linkedin.photon.ml.hyperparameter.EvaluationFunction
import com.linkedin.photon.ml.util.DoubleRange

/**
 * Test cases for the GaussianProcessSearch class
 */
class GaussianProcessSearchTest {

  val seed = 1L
  val dim = 3
  val n = 5
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

  val evaluator = new Evaluator {

    override protected[ml] val labelAndOffsetAndWeights: RDD[(Long, (Double, Double, Double))] =
      mock(classOf[RDD[(Long, (Double, Double, Double))]])
    override val evaluatorType = EvaluatorType.AUC
    override protected[ml] def evaluateWithScoresAndLabelsAndWeights(
        scoresAndLabelsAndWeights: RDD[(Long, (Double, Double, Double))]): Double = 0.0

    override def betterThan(score1: Double, score2: Double): Boolean = score1 > score2
  }

  val searcher = new GaussianProcessSearch[TestModel](ranges, evaluationFunction, evaluator, seed = seed)

  var observedPoints: Option[DenseMatrix[Double]] = None
  var observedEvals: Option[DenseVector[Double]] = None
  var bestEval: Double = evaluator.defaultScore
  var priorObservedPoints: Option[DenseMatrix[Double]] = None
  var priorObservedEvals: Option[DenseVector[Double]] = None
  var priorBestEval: Double = evaluator.defaultScore

  @Test
  def testFind(): Unit = {
    val candidates = searcher.find(n)

    assertEquals(candidates.length, n)
    assertTrue(candidates.forall(_.params.toArray.forall(x => x >= lower && x <= upper)))
    assertEquals(candidates.toSet.size, n)
  }

  @DataProvider
  def bestCandidateDataProvider: Array[Array[Any]] = {
    val candidate1 = DenseVector(1.0)
    val candidate2 = DenseVector(2.0)
    val candidate3 = DenseVector(3.0)
    val candidates = DenseMatrix.vertcat(
      candidate1.asDenseMatrix,
      candidate2.asDenseMatrix,
      candidate3.asDenseMatrix)

    Array(
      Array(candidates, DenseVector(2.0, 1.0, 0.0), candidate1),
      Array(candidates, DenseVector(1.0, 2.0, 0.0), candidate2),
      Array(candidates, DenseVector(0.0, 1.0, 2.0), candidate3))
  }

  @Test(dataProvider = "bestCandidateDataProvider")
  def testSelectBestCandidate(
      candidates: DenseMatrix[Double],
      predictions: DenseVector[Double],
      expected: DenseVector[Double]): Unit = {

    val selected = searcher.selectBestCandidate(candidates, predictions)
    assertEquals(selected, expected)
  }

  @DataProvider
  def priorDataProvider: Array[Array[Any]] = {

    val candidate1 = (DenseVector(1.0, 1.0, 1.0), 0.1)
    val candidate2 = (DenseVector(2.0, 2.0, 2.0), 0.2)
    val candidate3 = (DenseVector(3.0, 3.0, 3.0), 0.3)
    val currentCandidates = Seq(candidate1, candidate2, candidate3)
    val priorCandidates = Seq(candidate3, candidate1, candidate2)

    Array(
      Array(currentCandidates, None, 0),
      Array(currentCandidates, Option(priorCandidates), 1))
  }

  @Test(dataProvider = "priorDataProvider")
  def testOnObservation(
                         currentCandidates: Seq[(DenseVector[Double], Double)],
                         priorCandidates: Option[Seq[(DenseVector[Double], Double)]] = None,
                         testSetIndex: Int): Unit = {

    // Load the initial observations
    currentCandidates.foreach { case (candidate, value) =>
      val (newPoints, newEvals, newBest) = searcher.addObservation(observedPoints,
        observedEvals,
        bestEval,
        candidate,
        value)
      observedPoints = newPoints
      observedEvals = newEvals
      bestEval = newBest
    }

    if(priorCandidates.isDefined) {
      // Load the prior observations
      priorCandidates.get.foreach { case (candidate, value) =>
        val (newPoints, newEvals, newBest) = searcher.addObservation(priorObservedPoints,
          priorObservedEvals,
          priorBestEval,
          candidate,
          value)
        priorObservedPoints = newPoints
        priorObservedEvals = newEvals
        priorBestEval = newBest
      }
    }

    testSetIndex match {
      case 0 =>
        assertEquals(observedPoints.get.rows, 3)
        assertEquals(observedPoints.get.cols, 3)
        assertEquals(observedEvals.get.length, 3)
        assertEquals(bestEval, 0.3)
        assertFalse(priorObservedPoints.isDefined)
        assertFalse(priorObservedEvals.isDefined)
      case 1 =>
        assertEquals(observedPoints.get.rows, 6)
        assertEquals(observedPoints.get.cols, 3)
        assertEquals(observedEvals.get.length, 6)
        assertEquals(bestEval, 0.3)
        assertEquals(priorObservedPoints.get.rows, 3)
        assertEquals(priorObservedPoints.get.cols, 3)
        assertEquals(priorObservedEvals.get.length, 3)
        assertEquals(priorBestEval, 0.3)
    }
  }

  @Test(dataProvider = "priorDataProvider")
  def testFindWithPrior(
                         currentCandidates: Seq[(DenseVector[Double], Double)],
                         priorCandidates: Option[Seq[(DenseVector[Double], Double)]] = None,
                         testSetIndex: Int): Unit = {

    val candidates1 = searcher.findWithPrior(n, currentCandidates, priorCandidates)

    assertEquals(candidates1.length, n)
    assertTrue(candidates1.forall(_.params.toArray.forall(x => x >= lower && x <= upper)))
    assertEquals(candidates1.toSet.size, n)

  }
}
