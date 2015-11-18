/*
 * Copyright 2015 LinkedIn Corp. All rights reserved.
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
package com.linkedin.photon.ml.optimization

import java.util.Random

import breeze.linalg.{DenseVector, Vector, sum}
import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.function.DiffFunction
import com.linkedin.photon.ml.test.Assertions
import org.testng.Assert
import org.testng.annotations.{DataProvider, Test}


/**
 * Test that LBFGS/OWLQN can shrink the coefficients to zero.
 *
 * The objective function is (x-1)^2 + (y-4)^2, with L1 regularization C(|x| + |y|). We are only interested in the behavior around
 * the minimum, where x > 0 and y > 0. Thus the function to be optimized then become (x-1)^2 + (y-4)^2 + C(x+y) which has obvious analytic
 * solution. The test is based on this function and verifies the shrinkage of x and y.
 *
 * @author dpeng
 * @author nkatariy
 */

class LBFGSTest {
  val epsilon = 1.0E-6
  val random = new Random(1)

  private def getRandomInput(dim: Int): DenseVector[Double] = {
    DenseVector(Seq.fill(dim)(random.nextGaussian).toArray)
  }

  @Test(dataProvider = "dataProvider")
  def testLBFGS(l1Weight: Double, constraintMap: Option[Map[Int, (Double, Double)]],
                minimum: Array[Double], expectedValue: Double): Unit = {
    val lbfgs = new LBFGS[LabeledPoint]
    lbfgs.constraintMap = constraintMap

    var testFunc: DiffFunction[LabeledPoint] = null
    var dummyData: Array[LabeledPoint] = null
    var initCoeff: DenseVector[Double] = null

    constraintMap match {
      case Some(x) =>
        testFunc = if (l1Weight == 0) {
          new Test1DFunction
        } else {
          DiffFunction.withRegularization(new Test1DFunction, L1RegularizationContext, l1Weight)
        }
        dummyData = Array(LabeledPoint(0.0, DenseVector[Double](0.0), 0.0, 0.0))
        initCoeff = getRandomInput(1)
      case None =>
        testFunc = if (l1Weight == 0) {
          new Test2DFunction
        } else {
          DiffFunction.withRegularization(new Test2DFunction, L1RegularizationContext, l1Weight)
        }
        dummyData = Array(LabeledPoint(0.0, DenseVector[Double](0.0, 0.0), 0.0, 0.0))
        initCoeff = getRandomInput(2)
    }
    val (coef, actualValue) = lbfgs.optimize(dummyData, testFunc, initCoeff)
    Assertions.assertIterableEqualsWithTolerance(coef.toArray, minimum, epsilon)
    Assert.assertEquals(actualValue, expectedValue, epsilon)
  }

  @DataProvider(name = "dataProvider")
  def dataProvider(): Array[Array[Any]] = {
    Array(
      Array(0.0, None, Array(1.0, 4.0), 0.0),
      Array(1.0, None, Array(0.5, 3.5), 4.5),
      Array(2.0, None, Array(0.0, 3.0), 8.0),
      Array(8.0, None, Array(0.0, 0.0), 17.0),

      // note that expected value here is the value of the function at the unconstrained optima since the
      // projection happens after it
      Array(0.0, Some(Map[Int, (Double, Double)]()), Array(4.0), 0.0),
      Array(1.0, Some(Map[Int, (Double, Double)](0 -> (2.0, 3.0))), Array(3.0), 3.75),
      Array(2.0, Some(Map[Int, (Double, Double)](0 -> (-2.0, -1.0))), Array(-1.0), 7),
      Array(8.0, Some(Map[Int, (Double, Double)](0 -> (3.5, Double.PositiveInfinity))), Array(3.5), 16)
    )
  }
}

/**
 * A test function (x-1)**2 + (y-4)**2
 */
class Test2DFunction extends DiffFunction[LabeledPoint] {
  override def calculateAt(dataPoint: LabeledPoint, parameter: Vector[Double], cumGradient: Vector[Double]): Double = {
    val delta = parameter - Test2DFunction.centroid
    val deltaSq = delta.mapValues { x => x * x }
    cumGradient += delta :* 2.0
    sum(deltaSq)
  }
}

object Test2DFunction {
  val x0 = 1.0d
  val y0 = 4.0d
  val centroid = DenseVector[Double](x0, y0)
}

/**
 * A test function (x-4)**2
 */
class Test1DFunction extends DiffFunction[LabeledPoint] {
  override def calculateAt(dataPoint: LabeledPoint, parameter: Vector[Double], cumGradient: Vector[Double]): Double = {
    val delta = parameter - Test1DFunction.centroid
    val deltaSq = delta.mapValues { x => x * x }
    cumGradient += delta :* 2.0
    sum(deltaSq)
  }
}

object Test1DFunction {
  val x0 = 4.0d
  val centroid = DenseVector[Double](x0)
}