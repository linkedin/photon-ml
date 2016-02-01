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
package com.linkedin.photon.ml.function

import java.util.Random

import breeze.linalg.{DenseVector, SparseVector, Vector}
import com.linkedin.photon.ml.optimization._
import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.optimization._
import com.linkedin.photon.ml.test.SparkTestUtils
import org.apache.log4j.{LogManager, Logger}
import org.testng.Assert.{assertEquals, assertTrue}
import org.testng.annotations.{DataProvider, Test}

import scala.collection.mutable.ListBuffer


/**
 * Unit tests to check to verify the gradients / Hessians and other methods
 * @author bdrew
 * @author yali
 */

class ObjectiveFunctionTest extends SparkTestUtils {

  /**
   * List everything that conforms to DiffFunction here
   */
  @DataProvider(parallel = true)
  def getDifferentiableFunctions: Array[Array[Object]] = {
    // List of functions that return a tuple containing an undecorated loss function and its corresponding local data
    // set
    val baseLossFunctions = Array(
      () => ("Differentiable dummy objective to test optimizer, benign data", new TestObjective(), generateBenignLocalDataSetBinaryClassification()),
      () => ("Differentiable dummy objective to test optimizer, benign data, weighted examples", new TestObjective(), generateBenignLocalWeightedDataSetBinaryClassification()),
      () => ("Differentiable logistic loss, benign data", new LogisticLossFunction(), generateBenignLocalDataSetBinaryClassification()),
      () => ("Differentiable logistic loss, benign data, weighted examples", new LogisticLossFunction(), generateBenignLocalWeightedDataSetBinaryClassification()),
      () => ("Differentiable squared loss, benign data", new SquaredLossFunction(), generateBenignLocalDataSetBinaryClassification()),
      () => ("Differentiable squared loss, benign data, weighted examples", new SquaredLossFunction(), generateBenignLocalWeightedDataSetBinaryClassification()),
      () => ("Differentiable poisson loss, benign data", new PoissonLossFunction(), generateBenignLocalDataSetPoissonRegression()),
      () => ("Differentiable poisson loss, benign data, weighted examples", new PoissonLossFunction(), generateBenignLocalWeightedDataSetPoissonRegression()),
      () => ("Differentiable dummy objective to test optimizer, outlier data", new TestObjective(), generateOutlierLocalDataSetBinaryClassification()),
      () => ("Differentiable dummy objective to test optimizer, outlier data, weighted examples", new TestObjective(), generateOutlierLocalWeightedDataSetBinaryClassification()),
      () => ("Differentiable logistic loss, outlier data", new LogisticLossFunction(), generateOutlierLocalWeightedDataSetBinaryClassification()),
      () => ("Differentiable squared loss, outlier data, weighted examples", new SquaredLossFunction(), generateOutlierLocalWeightedDataSetBinaryClassification()),
      () => ("Differentiable poisson loss, outlier data", new PoissonLossFunction(), generateOutlierLocalDataSetPoissonRegression()),
      () => ("Differentiable poisson loss, outlier data, weighted examples", new PoissonLossFunction(), generateOutlierLocalWeightedDataSetPoissonRegression()))

    // List of regularization decorators. For each item in the base loss function list, we apply each decorator
    val regularizationDecorators = Array(
      (x:DiffFunction[LabeledPoint], baseDesc:String) => (s"$baseDesc with DiffFunction L2 regularization", DiffFunction.withRegularization(x, L2RegularizationContext, ObjectiveFunctionTest.REGULARIZATION_STRENGTH)),
      (x:DiffFunction[LabeledPoint], baseDesc:String) => (s"$baseDesc with DiffFunction L1 regularization", DiffFunction.withRegularization(x, L1RegularizationContext, ObjectiveFunctionTest.REGULARIZATION_STRENGTH)),
      (x:DiffFunction[LabeledPoint], baseDesc:String) => (s"$baseDesc with DiffFunction elastic net regularization", DiffFunction.withRegularization(x, new RegularizationContext(RegularizationType.ELASTIC_NET, Option(0.5)), ObjectiveFunctionTest.REGULARIZATION_STRENGTH)),
      (x:TwiceDiffFunction[LabeledPoint], baseDesc:String) => (s"$baseDesc with TwiceDiffFunction L2 regularization", TwiceDiffFunction.withRegularization(x, L2RegularizationContext, ObjectiveFunctionTest.REGULARIZATION_STRENGTH)),
      (x:TwiceDiffFunction[LabeledPoint], baseDesc:String) => (s"$baseDesc with TwiceDiffFunction L1 regularization", TwiceDiffFunction.withRegularization(x, L1RegularizationContext, ObjectiveFunctionTest.REGULARIZATION_STRENGTH)),
      (x:TwiceDiffFunction[LabeledPoint], baseDesc:String) => (s"$baseDesc with TwiceDiffFunction elastic net regularization", TwiceDiffFunction.withRegularization(x, new RegularizationContext(RegularizationType.ELASTIC_NET, Option(0.5)), ObjectiveFunctionTest.REGULARIZATION_STRENGTH))
    )

    val tmp:ListBuffer[(String, DiffFunction[LabeledPoint], Seq[LabeledPoint])] = scala.collection.mutable.ListBuffer()

    // Generate cartesian product of all regularization types by all base loss functions
    baseLossFunctions.map( { f =>
      val undecorated = f()
      tmp.append(undecorated)

      regularizationDecorators.map( { regularize =>
        val decorated = regularize(undecorated._2, undecorated._1)
        tmp.append( (decorated._1, decorated._2, undecorated._3))
      })
    })

    tmp.map( { x => Array(x._1, x._2, x._3) }).toArray
  }

  /**
   * List everything that conforms to TwiceDiffFunction here
   */
  @DataProvider(parallel = true)
  def getTwiceDifferentiableFunctions: Array[Array[Object]] = {
    val baseLossFunctions = Array(
      () => ("Differentiable logistic loss, benign data", new LogisticLossFunction(), generateBenignLocalDataSetBinaryClassification()),
      () => ("Differentiable logistic loss, benign data, weighted examples", new LogisticLossFunction(), generateBenignLocalWeightedDataSetBinaryClassification()),
      () => ("Differentiable squared loss, benign data", new SquaredLossFunction(), generateBenignLocalDataSetBinaryClassification()),
      () => ("Differentiable squared loss, benign data, weighted examples", new SquaredLossFunction(), generateBenignLocalWeightedDataSetBinaryClassification()),
      () => ("Differentiable poisson loss, benign data", new PoissonLossFunction(), generateBenignLocalDataSetPoissonRegression()),
      () => ("Differentiable poisson loss, benign data, weighted examples", new PoissonLossFunction(), generateBenignLocalWeightedDataSetPoissonRegression()),
      () => ("Differentiable logistic loss, outlier data", new LogisticLossFunction(), generateOutlierLocalDataSetBinaryClassification()),
      () => ("Differentiable logistic loss, outlier data, weighted examples", new LogisticLossFunction(), generateOutlierLocalWeightedDataSetBinaryClassification()),
      () => ("Differentiable squared loss, outlier data", new SquaredLossFunction(), generateOutlierLocalDataSetBinaryClassification()),
      () => ("Differentiable squared loss, outlier data, weighted examples", new SquaredLossFunction(), generateOutlierLocalWeightedDataSetBinaryClassification()),
      () => ("Differentiable poisson loss, outlier data", new PoissonLossFunction(), generateOutlierLocalDataSetPoissonRegression()),
      () => ("Differentiable poisson loss, outlier data, weighted examples", new PoissonLossFunction(), generateOutlierLocalWeightedDataSetPoissonRegression()))

    // List of regularization decorators. For each item in the base loss function list, we apply each decorator
    val regularizationDecorators = Array(
      (x:TwiceDiffFunction[LabeledPoint], baseDesc:String) => (s"$baseDesc with TwiceDiffFunction L2 regularization", TwiceDiffFunction.withRegularization(x, L2RegularizationContext, ObjectiveFunctionTest.REGULARIZATION_STRENGTH)),
      (x:TwiceDiffFunction[LabeledPoint], baseDesc:String) => (s"$baseDesc with TwiceDiffFunction L1 regularization", TwiceDiffFunction.withRegularization(x, L1RegularizationContext, ObjectiveFunctionTest.REGULARIZATION_STRENGTH)),
      (x:TwiceDiffFunction[LabeledPoint], baseDesc:String) => (s"$baseDesc with TwiceDiffFunction Elastic Net regularization", TwiceDiffFunction.withRegularization(x, new RegularizationContext(RegularizationType.ELASTIC_NET, Option(0.5)), ObjectiveFunctionTest.REGULARIZATION_STRENGTH))
    )

    val tmp:scala.collection.mutable.ListBuffer[(String, DiffFunction[LabeledPoint], Seq[LabeledPoint])] = scala.collection.mutable.ListBuffer()

    // Generate cartesian product of all regularization types by all base loss functions
    baseLossFunctions.map( { f =>
      val undecorated = f()
      tmp.append(undecorated)

      regularizationDecorators.map( { regularize =>
        val decorated = regularize(undecorated._2, undecorated._1)
        tmp.append( (decorated._1, decorated._2, undecorated._3))
      })
    })

    tmp.map( { x => Array(x._1, x._2, x._3) }).toArray
  }

  def generateBenignLocalDataSetBinaryClassification(): List[LabeledPoint] = {
    drawBalancedSampleFromNumericallyBenignDenseFeaturesForBinaryClassifierLocal(
      ObjectiveFunctionTest.DATA_RANDOM_SEED,
      ObjectiveFunctionTest.TRAINING_SAMPLES,
      ObjectiveFunctionTest.PROBLEM_DIMENSION).map({ obj =>
        assertEquals(obj._2.length, ObjectiveFunctionTest.PROBLEM_DIMENSION, "Samples should have expected lengths")
        new LabeledPoint(label = obj._1, obj._2, offset = 0, weight = 1)
      }).toList
  }


  def generateBenignLocalWeightedDataSetBinaryClassification(): List[LabeledPoint] = {
    val r: Random = new Random(ObjectiveFunctionTest.WEIGHT_RANDOM_SEED)
    drawBalancedSampleFromNumericallyBenignDenseFeaturesForBinaryClassifierLocal(
      ObjectiveFunctionTest.DATA_RANDOM_SEED,
      ObjectiveFunctionTest.TRAINING_SAMPLES,
      ObjectiveFunctionTest.PROBLEM_DIMENSION).map({ obj =>
        assertEquals(obj._2.length, ObjectiveFunctionTest.PROBLEM_DIMENSION, "Samples should have expected lengths")
        val weight: Double = r.nextDouble() * ObjectiveFunctionTest.WEIGHT_RANDOM_MAX
        new LabeledPoint(label = obj._1, obj._2, offset = 0, weight)
      }).toList
  }

  def generateOutlierLocalDataSetBinaryClassification(): List[LabeledPoint] = {
    drawBalancedSampleFromOutlierDenseFeaturesForBinaryClassifierLocal(
      ObjectiveFunctionTest.DATA_RANDOM_SEED,
      ObjectiveFunctionTest.TRAINING_SAMPLES,
      ObjectiveFunctionTest.PROBLEM_DIMENSION).map({ obj =>
      assertEquals(obj._2.length, ObjectiveFunctionTest.PROBLEM_DIMENSION, "Samples should have expected lengths")
      new LabeledPoint(label = obj._1, obj._2, offset = 0, weight = 1)
    }).toList
  }

  def generateOutlierLocalWeightedDataSetBinaryClassification(): List[LabeledPoint] = {
    val r: Random = new Random(ObjectiveFunctionTest.WEIGHT_RANDOM_SEED)
    drawBalancedSampleFromOutlierDenseFeaturesForBinaryClassifierLocal(
      ObjectiveFunctionTest.DATA_RANDOM_SEED,
      ObjectiveFunctionTest.TRAINING_SAMPLES,
      ObjectiveFunctionTest.PROBLEM_DIMENSION).map({ obj =>
      assertEquals(obj._2.length, ObjectiveFunctionTest.PROBLEM_DIMENSION, "Samples should have expected lengths")
        val weight: Double = r.nextDouble() * ObjectiveFunctionTest.WEIGHT_RANDOM_MAX
        new LabeledPoint(label = obj._1, obj._2, offset = 0, weight)
      }).toList
  }

  def generateBenignLocalDataSetPoissonRegression(): List[LabeledPoint] = {
    drawSampleFromNumericallyBenignDenseFeaturesForPoissonRegressionLocal(
      ObjectiveFunctionTest.DATA_RANDOM_SEED,
      ObjectiveFunctionTest.TRAINING_SAMPLES,
      ObjectiveFunctionTest.PROBLEM_DIMENSION).map({ obj =>
        assertEquals(obj._2.length, ObjectiveFunctionTest.PROBLEM_DIMENSION, "Samples should have expected lengths")
        new LabeledPoint(label = obj._1, obj._2, offset = 0, weight = 1)
    }).toList
  }

  def generateBenignLocalWeightedDataSetPoissonRegression(): List[LabeledPoint] = {
    val r: Random = new Random(ObjectiveFunctionTest.WEIGHT_RANDOM_SEED)
    drawSampleFromNumericallyBenignDenseFeaturesForPoissonRegressionLocal(
      ObjectiveFunctionTest.DATA_RANDOM_SEED,
      ObjectiveFunctionTest.TRAINING_SAMPLES,
      ObjectiveFunctionTest.PROBLEM_DIMENSION).map({ obj =>
        assertEquals(obj._2.length, ObjectiveFunctionTest.PROBLEM_DIMENSION, "Samples should have expected lengths")
        val weight: Double = r.nextDouble() * ObjectiveFunctionTest.WEIGHT_RANDOM_MAX
        new LabeledPoint(label = obj._1, obj._2, offset = 0, weight)
      }).toList
  }

  def generateOutlierLocalDataSetPoissonRegression(): List[LabeledPoint] = {
    drawSampleFromOutlierDenseFeaturesForPoissonRegressionLocal(
      ObjectiveFunctionTest.DATA_RANDOM_SEED,
      ObjectiveFunctionTest.TRAINING_SAMPLES,
      ObjectiveFunctionTest.PROBLEM_DIMENSION).map({ obj =>
        assertEquals(obj._2.length, ObjectiveFunctionTest.PROBLEM_DIMENSION, "Samples should have expected lengths")
        new LabeledPoint(label = obj._1, obj._2, offset = 0, weight = 1)
    }).toList
  }

  def generateOutlierLocalWeightedDataSetPoissonRegression(): List[LabeledPoint] = {
    val r: Random = new Random(ObjectiveFunctionTest.WEIGHT_RANDOM_SEED)
    drawSampleFromOutlierDenseFeaturesForPoissonRegressionLocal(
      ObjectiveFunctionTest.DATA_RANDOM_SEED,
      ObjectiveFunctionTest.TRAINING_SAMPLES,
      ObjectiveFunctionTest.PROBLEM_DIMENSION).map({ obj =>
        assertEquals(obj._2.length, ObjectiveFunctionTest.PROBLEM_DIMENSION, "Samples should have expected lengths")
        val weight: Double = r.nextDouble() * ObjectiveFunctionTest.WEIGHT_RANDOM_MAX
        new LabeledPoint(label = obj._1, obj._2, offset = 0, weight)
      }).toList
  }

  def checkGradient(prefix: String, objectiveBefore: Double, objectiveAfter: Double, expected: Double) = {
    val numericDeriv = (objectiveAfter - objectiveBefore) / (2.0 * ObjectiveFunctionTest.DERIVATIVE_DELTA)
    val relativeErrorNum = Math.abs(numericDeriv - expected)
    val relativeErrorFactor = Math.min(Math.abs(numericDeriv), Math.abs(expected))
    val relativeErrorDen = if (relativeErrorFactor > 0) { relativeErrorFactor } else { 1 }
    val relativeError = relativeErrorNum / relativeErrorDen

    assertTrue(java.lang.Double.isFinite(objectiveBefore), s"Objective before step [$objectiveBefore] should be finite")
    assertTrue(java.lang.Double.isFinite(objectiveAfter), s"Objective after step [$objectiveAfter] should be finite")

    if( !(numericDeriv.isInfinite || numericDeriv.isNaN) ) {
      assertTrue(
        relativeError < ObjectiveFunctionTest.GRADIENT_TOLERANCE ||
          relativeErrorNum < ObjectiveFunctionTest.GRADIENT_TOLERANCE,
        "Computed gradient and numerical differentiation estimate should be close." +
          s"$prefix estimated [$numericDeriv] " +
          s"v. computed [$expected] with absolute error [$relativeErrorNum] and " +
          s"relative error [$relativeError]")
    }
  }

  def checkHessian(prefix: String, gradBefore: Double, gradAfter: Double, expected: Double) = {
    val numericDeriv = (gradAfter - gradBefore) / (2.0 * ObjectiveFunctionTest.DERIVATIVE_DELTA)
    val relativeErrorNum = Math.abs(numericDeriv - expected)
    val relativeErrorFactor = Math.min(Math.abs(numericDeriv), Math.abs(expected))
    val relativeErrorDen = if (relativeErrorFactor > 0) { relativeErrorFactor } else { 1 }
    val relativeError = relativeErrorNum / relativeErrorDen

    assertTrue(
      relativeError < ObjectiveFunctionTest.HESSIAN_TOLERANCE ||
        relativeErrorNum < ObjectiveFunctionTest.HESSIAN_TOLERANCE,
      "Computed Hessian and numerical differentiation estimate should be close." +
        s"$prefix estimated [$numericDeriv] " +
        s"v. computed [$expected] with absolute error [$relativeErrorNum] and " +
        s"relative error [$relativeError]")
  }


  /**
   * Verify that gradient is consistent with the objective when computed locally
   */
  @Test(dataProvider = "getDifferentiableFunctions",
        groups = Array[String]("ObjectiveFunctionTests", "testCore"))
  def checkGradientConsistentWithObjectiveLocal(description:String, function: DiffFunction[LabeledPoint], data:Seq[LabeledPoint]): Unit = {
    val r: Random = new Random(ObjectiveFunctionTest.PARAMETER_RANDOM_SEED)

    for (iter <- 0 until ObjectiveFunctionTest.LOCAL_CONSISTENCY_CHECK_SAMPLES) {
      val initParam: Vector[Double] = DenseVector.fill[Double](ObjectiveFunctionTest.PROBLEM_DIMENSION) { if (iter > 0) { r.nextDouble() } else { 0 } }
      val computed = function.calculate(data, initParam)

      // Element-wise numerical differentiation to get the gradient
      for (idx <- 0 until ObjectiveFunctionTest.PROBLEM_DIMENSION) {
        val before = initParam.copy
        before(idx) -= ObjectiveFunctionTest.DERIVATIVE_DELTA
        val after = initParam.copy
        after(idx) += ObjectiveFunctionTest.DERIVATIVE_DELTA
        val objBefore = function.calculate(data, before)._1
        val objAfter = function.calculate(data, after)._1
        checkGradient(
          s" f=[$function / ${function.getClass.getName}], iter=[$iter], idx=[$idx] ",
          objBefore,
          objAfter,
          computed._2(idx))
      }
    }
  }

  /**
   * Verify that the Hessian is consistent with the gradient when computed locally
   */
  @Test(dataProvider = "getTwiceDifferentiableFunctions",
        dependsOnMethods = Array[String]("checkGradientConsistentWithObjectiveLocal"),
        groups = Array[String]("ObjectiveFunctionTests", "testCore"))
  def checkHessianConsistentWithObjectiveLocal(description:String, function: TwiceDiffFunction[LabeledPoint], data: Seq[LabeledPoint]) = {
    val r: Random = new Random(ObjectiveFunctionTest.PARAMETER_RANDOM_SEED)

    for (iter <- 0 until ObjectiveFunctionTest.LOCAL_CONSISTENCY_CHECK_SAMPLES) {
      val initParam: Vector[Double] = DenseVector.fill[Double](ObjectiveFunctionTest.PROBLEM_DIMENSION) { if (iter > 0) { r.nextDouble() } else { 0 } }

      // Loop over basis vectors. This will give us H*e_i = H(:,i) (so one column of H at a time)
      for (basis <- 0 until ObjectiveFunctionTest.PROBLEM_DIMENSION) {
        val basisVector: Vector[Double] = new SparseVector[Double](Array[Int](basis), Array[Double](1.0), 1, ObjectiveFunctionTest.PROBLEM_DIMENSION)
        val hessianVector = function.hessianVector(data, initParam, basisVector)

        // Element-wise numerical differentiation to get the Hessian
        for (idx <- 0 until ObjectiveFunctionTest.PROBLEM_DIMENSION) {
          val before = initParam.copy
          before(idx) -= ObjectiveFunctionTest.DERIVATIVE_DELTA
          val after = initParam.copy
          after(idx) += ObjectiveFunctionTest.DERIVATIVE_DELTA
          val gradBefore = function.gradient(data, before)
          val gradAfter = function.gradient(data, after)
          checkHessian(
            s"Iteration [$iter], basis=[$basis], idx=[$idx], Hessian=[$hessianVector]",
            gradBefore(basis),
            gradAfter(basis),
            hessianVector(idx))
        }
      }
    }
  }
}

object ObjectiveFunctionTest {
  val LOCAL_CONSISTENCY_CHECK_SAMPLES = 1000
  val PROBLEM_DIMENSION: Int = 10
  val REGULARIZATION_STRENGTH: Double = 100
  val DERIVATIVE_DELTA: Double = 1e-6
  val GRADIENT_TOLERANCE: Double = 1e-3
  val HESSIAN_TOLERANCE: Double = 1e-3
  val DATA_RANDOM_SEED: Int = 0
  val WEIGHT_RANDOM_SEED: Int = 100
  val WEIGHT_RANDOM_MAX: Double = 10.0
  val PARAMETER_RANDOM_SEED: Int = 500
  val TRAINING_SAMPLES = PROBLEM_DIMENSION * PROBLEM_DIMENSION
  val LOGGER: Logger = LogManager.getLogger(classOf[ObjectiveFunctionTest])
}
