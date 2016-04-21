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
import org.testng.annotations.{DataProvider, Test}

/**
 * @author nkatariy
 */
class OptimizationProblemTest {
  case class Mocks (optimizer: AbstractOptimizer[LabeledPoint, TwiceDiffFunction[LabeledPoint]],
                    objectiveFunction: TwiceDiffFunction[LabeledPoint],
                    rdd: RDD[LabeledPoint],
                    sparkContext: SparkContext,
                    broadcastVector: Broadcast[Vector[Double]],
                    iterable: Iterable[LabeledPoint],
                    coeffs: Coefficients,
                    vector: Vector[Double],
                    variancesOption: Option[Vector[Double]])

  def getMocks: Mocks = {
    Mocks(
      Mockito.mock(classOf[AbstractOptimizer[LabeledPoint, TwiceDiffFunction[LabeledPoint]]]),
      Mockito.mock(classOf[TwiceDiffFunction[LabeledPoint]]),
      Mockito.mock(classOf[RDD[LabeledPoint]]),
      Mockito.mock(classOf[SparkContext]),
      Mockito.mock(classOf[Broadcast[Vector[Double]]]),
      Mockito.mock(classOf[Iterable[LabeledPoint]]),
      Mockito.mock(classOf[Coefficients]),
      Mockito.mock(classOf[Vector[Double]]),
      Mockito.mock(classOf[Option[Vector[Double]]])
    )
  }

  def getProblem(optimizer: AbstractOptimizer[LabeledPoint, TwiceDiffFunction[LabeledPoint]],
                 objectiveFunction: TwiceDiffFunction[LabeledPoint],
                 regularizationWeight: Double = 0.5) = {
    val mockLossFunction = Mockito.mock(classOf[TwiceDiffFunction[LabeledPoint]])
    val mockSampler = Mockito.mock(classOf[DownSampler])

    new OptimizationProblem(optimizer, objectiveFunction, mockLossFunction,
      regularizationWeight, mockSampler)
  }

  @DataProvider
  def regularizationTermTestData(): Array[Array[Any]] = {
    val mocks = getMocks
    val mockOptimizer = mocks.optimizer
    val mockObjectiveFunction = mocks.objectiveFunction
    val mockCoeffs = mocks.coeffs
    val mockVector = mocks.vector

    val dummyDotProduct = 20.0
    Mockito.when(mockCoeffs.means).thenReturn(mockVector)
    Mockito.when(mockVector.dot(mockVector)).thenReturn(dummyDotProduct)

    Array(
      Array(mockOptimizer, mockObjectiveFunction, 0.1, mockCoeffs, 1.0),
      Array(mockOptimizer, mockObjectiveFunction, 1, mockCoeffs, 10.0),
      Array(mockOptimizer, mockObjectiveFunction, 10.0, mockCoeffs, 100.0)
    )
  }

  @Test(dataProvider = "regularizationTermTestData")
  def testGetRegularizationTermValue(mockOptimizer: AbstractOptimizer[LabeledPoint, TwiceDiffFunction[LabeledPoint]],
                                     mockObjectiveFunction: TwiceDiffFunction[LabeledPoint],
                                     regWeight: Double,
                                     coeffs: Coefficients,
                                     expectedRegularizationTermValue: Double) = {
    Assert.assertEquals(getProblem(mockOptimizer, mockObjectiveFunction, regWeight).getRegularizationTermValue(coeffs),
      expectedRegularizationTermValue)
  }

  @Test
  def testUpdateCoefficientVariancesIterable() = {
    val mocks = getMocks

    val hessianDiagonalOutput = new DenseVector[Double](Array(0.3, -0.5, 0.0, 0.1))
    val expectedVariances = Array[Double](10.0 / 3, -2.0, 1 / MathConst.HIGH_PRECISION_TOLERANCE_THRESHOLD, 10.0)

    Mockito.when(mocks.objectiveFunction.hessianDiagonal(Matchers.any(classOf[Iterable[LabeledPoint]]), Matchers.any(classOf[Vector[Double]])))
      .thenReturn(hessianDiagonalOutput)
    val problem = getProblem(mocks.optimizer, mocks.objectiveFunction)

    Mockito.when(mocks.coeffs.means).thenReturn(mocks.vector)

    val result = problem.updateCoefficientsVariances(mocks.iterable, mocks.coeffs)
    Assert.assertEquals(result.means, mocks.vector)
    result.variancesOption.foreach(x => x.toArray.zip(expectedVariances)
      .foreach(y => Assert.assertEquals(y._1, y._2, MathConst.MEDIUM_PRECISION_TOLERANCE_THRESHOLD)))
  }

  @Test
  def testUpdateCoefficientMeansIterable() = {
    val mocks = getMocks

    Mockito.when(mocks.coeffs.means).thenReturn(mocks.vector)
    Mockito.when(mocks.coeffs.variancesOption).thenReturn(mocks.variancesOption)

    val mockUpdatedCoeffs = Mockito.mock(classOf[Vector[Double]])
    val mockLossValue = Matchers.anyDouble()
    Mockito.when(mocks.optimizer.optimize(Matchers.any(classOf[Iterable[LabeledPoint]]),
      Matchers.any(classOf[TwiceDiffFunction[LabeledPoint]]), mocks.vector))
      .thenReturn((mockUpdatedCoeffs, mockLossValue))

    val problem = getProblem(mocks.optimizer, mocks.objectiveFunction)
    val result = problem.updateCoefficientMeans(mocks.iterable, mocks.coeffs)
    Assert.assertEquals(result._1.means, mockUpdatedCoeffs)
    Assert.assertEquals(result._1.variancesOption, mocks.variancesOption)
    Assert.assertEquals(result._2, mockLossValue)
  }

  @Test
  def testUpdateCoefficientVariancesRDD() = {
    val mocks = getMocks

    Mockito.when(mocks.rdd.sparkContext).thenReturn(mocks.sparkContext)
    Mockito.when(mocks.sparkContext.broadcast(mocks.vector)).thenReturn(mocks.broadcastVector)

    val hessianDiagonalOutput = new DenseVector[Double](Array(0.3, -0.5, 0.0, 0.1))
    val expectedVariances = Array[Double](10.0 / 3, -2.0, 1 / MathConst.HIGH_PRECISION_TOLERANCE_THRESHOLD, 10.0)

    Mockito.when(mocks.objectiveFunction.hessianDiagonal(Matchers.any(classOf[RDD[LabeledPoint]]), Matchers.any(classOf[Broadcast[Vector[Double]]])))
      .thenReturn(hessianDiagonalOutput)
    val problem = getProblem(mocks.optimizer, mocks.objectiveFunction)

    Mockito.when(mocks.coeffs.means).thenReturn(mocks.vector)

    val result = problem.updateCoefficientsVariances(mocks.rdd, mocks.coeffs)
    Assert.assertEquals(result.means, mocks.vector)
    result.variancesOption.foreach(x => x.toArray.zip(expectedVariances)
      .foreach(y => Assert.assertEquals(y._1, y._2, MathConst.MEDIUM_PRECISION_TOLERANCE_THRESHOLD)))
  }

  @Test
  def testUpdateCoefficientMeansRDD() = {
    val mocks = getMocks

    Mockito.when(mocks.coeffs.means).thenReturn(mocks.vector)
    Mockito.when(mocks.coeffs.variancesOption).thenReturn(mocks.variancesOption)

    val mockUpdatedCoeffs = Mockito.mock(classOf[Vector[Double]])
    val mockLossValue = Matchers.anyDouble()
    Mockito.when(mocks.optimizer.optimize(Matchers.any(classOf[RDD[LabeledPoint]]),
      Matchers.any(classOf[TwiceDiffFunction[LabeledPoint]]), mocks.vector))
      .thenReturn((mockUpdatedCoeffs, mockLossValue))

    val problem = getProblem(mocks.optimizer, mocks.objectiveFunction)
    val result = problem.updateCoefficientMeans(mocks.rdd, mocks.coeffs)
    Assert.assertEquals(result._1.means, mockUpdatedCoeffs)
    Assert.assertEquals(result._1.variancesOption, mocks.variancesOption)
    Assert.assertEquals(result._2, mockLossValue)
  }
}