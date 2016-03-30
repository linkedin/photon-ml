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
package com.linkedin.photon.ml.optimization.game

import breeze.linalg.{DenseVector, Vector}
import com.linkedin.photon.ml.constants.MathConst
import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.function.TwiceDiffFunction
import com.linkedin.photon.ml.model.Coefficients
import com.linkedin.photon.ml.optimization.AbstractOptimizer
import com.linkedin.photon.ml.sampler.DownSampler
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.mockito.{Matchers, Mockito}
import org.testng.Assert
import org.testng.annotations.{DataProvider, BeforeTest, Test}

/**
 * @author nkatariy
 */
class OptimizationProblemTest {
  val mockOptimizer = Mockito.mock(classOf[AbstractOptimizer[LabeledPoint, TwiceDiffFunction[LabeledPoint]]])
  val mockObjectiveFunction =  Mockito.mock(classOf[TwiceDiffFunction[LabeledPoint]])

  val mockRDD = Mockito.mock(classOf[RDD[LabeledPoint]])
  val mockSparkContext = Mockito.mock(classOf[SparkContext])
  val mockBroadcastVector = Mockito.mock(classOf[Broadcast[Vector[Double]]])
  val mockIterable = Mockito.mock(classOf[Iterable[LabeledPoint]])
  val mockCoeffs = Mockito.mock(classOf[Coefficients])
  val mockVector = Mockito.mock(classOf[Vector[Double]])
  val mockVariancesOption = Mockito.mock(classOf[Option[Vector[Double]]])

  def getProblem(optimizer: AbstractOptimizer[LabeledPoint, TwiceDiffFunction[LabeledPoint]] = mockOptimizer,
                 objectiveFunction: TwiceDiffFunction[LabeledPoint] = mockObjectiveFunction,
                 regularizationWeight: Double = 0.5) = {
    val mockLossFunction = Mockito.mock(classOf[TwiceDiffFunction[LabeledPoint]])
    val mockSampler = Mockito.mock(classOf[DownSampler])

    new OptimizationProblem(optimizer, objectiveFunction, mockLossFunction,
      regularizationWeight, mockSampler)
  }

  @DataProvider
  def regularizationTermTestData(): Array[Array[Any]] = {
    val dummyDotProduct = 20.0
    Mockito.when(mockCoeffs.means).thenReturn(mockVector)
    Mockito.when(mockVector.dot(mockVector)).thenReturn(dummyDotProduct)

    Array(
      Array(0.1, mockCoeffs, 1.0),
      Array(1, mockCoeffs, 10.0),
      Array(10.0, mockCoeffs, 100.0)
    )
  }

  @Test(dataProvider = "regularizationTermTestData")
  def testGetRegularizationTermValue(regWeight: Double, coeffs: Coefficients, expectedRegularizationTermValue: Double) = {
    Assert.assertEquals(getProblem(regularizationWeight = regWeight).getRegularizationTermValue(coeffs),
      expectedRegularizationTermValue)
  }

  @Test
  def testUpdateCoefficientVariancesIterable() = {
    val hessianDiagonalOutput = new DenseVector[Double](Array(0.3, -0.5, 0.0, 0.1))
    val expectedVariances = Array[Double](10.0 / 3, -2.0, 1 / MathConst.HIGH_PRECISION_TOLERANCE_THRESHOLD, 10.0)

    Mockito.when(mockObjectiveFunction.hessianDiagonal(Matchers.any(classOf[Iterable[LabeledPoint]]), Matchers.any(classOf[Vector[Double]])))
      .thenReturn(hessianDiagonalOutput)
    val problem = getProblem(objectiveFunction = mockObjectiveFunction)

    Mockito.when(mockCoeffs.means).thenReturn(mockVector)

    val result = problem.updateCoefficientsVariances(mockIterable, mockCoeffs)
    Assert.assertEquals(result.means, mockVector)
    result.variancesOption.foreach(x => x.toArray.zip(expectedVariances)
      .foreach(y => Assert.assertEquals(y._1, y._2, MathConst.MEDIUM_PRECISION_TOLERANCE_THRESHOLD)))
  }

  @Test
  def testUpdateCoefficientMeansIterable() = {
    Mockito.when(mockCoeffs.means).thenReturn(mockVector)
    Mockito.when(mockCoeffs.variancesOption).thenReturn(mockVariancesOption)

    val mockUpdatedCoeffs = Mockito.mock(classOf[Vector[Double]])
    val mockLossValue = Matchers.anyDouble()
    Mockito.when(mockOptimizer.optimize(Matchers.any(classOf[Iterable[LabeledPoint]]),
      Matchers.any(classOf[TwiceDiffFunction[LabeledPoint]]), mockVector))
      .thenReturn((mockUpdatedCoeffs, mockLossValue))

    val problem = getProblem(optimizer = mockOptimizer)
    val result = problem.updateCoefficientMeans(mockIterable, mockCoeffs)
    Assert.assertEquals(result._1.means, mockUpdatedCoeffs)
    Assert.assertEquals(result._1.variancesOption, mockVariancesOption)
    Assert.assertEquals(result._2, mockLossValue)
  }

  @Test
  def testUpdateCoefficientVariancesRDD() = {
    Mockito.when(mockRDD.sparkContext).thenReturn(mockSparkContext)
    Mockito.when(mockSparkContext.broadcast(mockVector)).thenReturn(mockBroadcastVector)

    val hessianDiagonalOutput = new DenseVector[Double](Array(0.3, -0.5, 0.0, 0.1))
    val expectedVariances = Array[Double](10.0 / 3, -2.0, 1 / MathConst.HIGH_PRECISION_TOLERANCE_THRESHOLD, 10.0)

    val mockObjectiveFunction =  Mockito.mock(classOf[TwiceDiffFunction[LabeledPoint]])
    Mockito.when(mockObjectiveFunction.hessianDiagonal(Matchers.any(classOf[RDD[LabeledPoint]]), Matchers.any(classOf[Broadcast[Vector[Double]]])))
      .thenReturn(hessianDiagonalOutput)
    val problem = getProblem(objectiveFunction = mockObjectiveFunction)

    Mockito.when(mockCoeffs.means).thenReturn(mockVector)

    val result = problem.updateCoefficientsVariances(mockRDD, mockCoeffs)
    Assert.assertEquals(result.means, mockVector)
    result.variancesOption.foreach(x => x.toArray.zip(expectedVariances)
      .foreach(y => Assert.assertEquals(y._1, y._2, MathConst.MEDIUM_PRECISION_TOLERANCE_THRESHOLD)))
  }

  @Test
  def testUpdateCoefficientMeansRDD() = {
    Mockito.when(mockCoeffs.means).thenReturn(mockVector)
    Mockito.when(mockCoeffs.variancesOption).thenReturn(mockVariancesOption)

    val mockUpdatedCoeffs = Mockito.mock(classOf[Vector[Double]])
    val mockLossValue = Matchers.anyDouble()
    Mockito.when(mockOptimizer.optimize(Matchers.any(classOf[RDD[LabeledPoint]]),
      Matchers.any(classOf[TwiceDiffFunction[LabeledPoint]]), mockVector))
      .thenReturn((mockUpdatedCoeffs, mockLossValue))

    val problem = getProblem(optimizer = mockOptimizer)
    val result = problem.updateCoefficientMeans(mockRDD, mockCoeffs)
    Assert.assertEquals(result._1.means, mockUpdatedCoeffs)
    Assert.assertEquals(result._1.variancesOption, mockVariancesOption)
    Assert.assertEquals(result._2, mockLossValue)
  }
}