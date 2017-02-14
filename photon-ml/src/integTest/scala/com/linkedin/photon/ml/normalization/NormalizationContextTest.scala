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
package com.linkedin.photon.ml.normalization

import org.apache.spark.rdd.RDD

import org.testng.Assert._
import org.testng.annotations.{DataProvider, Test}

import breeze.linalg.{DenseVector, SparseVector, Vector}

import com.linkedin.photon.ml.TaskType
import com.linkedin.photon.ml.TaskType.TaskType
import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.function.glm.{DistributedGLMLossFunction, LogisticLossFunction, PoissonLossFunction, SquaredLossFunction}
import com.linkedin.photon.ml.optimization.OptimizerType.OptimizerType
import com.linkedin.photon.ml.optimization.{GLMOptimizationConfiguration, LBFGS, OptimizerType, TRON}
import com.linkedin.photon.ml.stat.BasicStatistics
import com.linkedin.photon.ml.test.Assertions.assertIterableEqualsWithTolerance
import com.linkedin.photon.ml.test.SparkTestUtils

/**
  * Test building NormalizationContext from summary. A sophisticated test with the heart data set is also performed to
  * verify that the standardization is correct numerically.
  */
class NormalizationContextTest extends SparkTestUtils {
  /*
   * features:
   *  1  1  1  1  1  0
   *  2  0 -1  0  1  0
   * .2  0 .5  0  1  0
   *  0 10  0  5  1  0
   *
   *  The fifth and sixth columns should be left untouched in all scalers.
   */
  private val _dim: Int = 6
  private val _intercept: Option[Int] = Some(4)
  private val _features = Array(
    DenseVector[Double](1.0, 1.0, 1.0, 1.0, 1.0, 0.0),
    DenseVector[Double](2.0, 0.0, -1.0, 0.0, 1.0, 0.0),
    new SparseVector[Double](Array(0, 2, 4), Array(0.2, 0.5, 1.0), _dim),
    new SparseVector[Double](Array(1, 3, 4), Array(10.0, 5.0, 1.0), _dim)
  )
  private val _delta = 1.0E-5
  private val _stdFactors = Array(1.09985, 0.20592, 1.17108, 0.42008, 1.0, 1.0)
  private val _maxMagnitudeFactors = Array(0.5, 0.1, 1.0, 0.2, 1.0, 1.0)
  private val _meanShifts = Array(0.8, 2.75, 0.125, 1.5, 0.0, 0.0)

  @Test(dataProvider = "generateTestData")
  def testFactor(
      normalizationType: NormalizationType,
      expectedFactors: Option[Array[Double]],
      expectedShifts: Option[Array[Double]]): Unit = sparkTest("test") {

    val rdd = sc.parallelize(_features.map(x => new LabeledPoint(0, x)))
    lazy val summary = BasicStatistics.getBasicStatistics(rdd)
    val normalizationContext = NormalizationContext(normalizationType, summary, _intercept)
    val factors = normalizationContext.factors
    (factors, expectedFactors) match {
      case (Some(_), None) | (None, Some(_)) =>
        throw new AssertionError("Calculated factors and expected factors don't match.")
      case (Some(f1), Some(f2)) =>
        assertIterableEqualsWithTolerance(f1.toArray, f2, _delta)
      case (None, None) => ;
    }
    val shifts = normalizationContext.shifts
    (shifts, expectedShifts) match {
      case (Some(_), None) | (None, Some(_)) =>
        throw new AssertionError("Calculated shifts and expected shifts don't match.")
      case (Some(s1), Some(s2)) =>
        assertIterableEqualsWithTolerance(s1.toArray, s2, _delta)
      case (None, None) => ;
    }
  }

  @DataProvider
  def generateTestData(): Array[Array[Any]] = {
    Array(
      Array(NormalizationType.SCALE_WITH_STANDARD_DEVIATION, Some(_stdFactors), None),
      Array(NormalizationType.SCALE_WITH_MAX_MAGNITUDE, Some(_maxMagnitudeFactors), None),
      Array(NormalizationType.NONE, None, None),
      Array(NormalizationType.STANDARDIZATION, Some(_stdFactors), Some(_meanShifts))
    )
  }

  @DataProvider(name = "generateStandardizationTestData")
  def generateStandardizationTestData(): Array[Array[Any]] = {
    (for (x <- OptimizerType.values;
          y <- TaskType.values.filter { t => t != TaskType.SMOOTHED_HINGE_LOSS_LINEAR_SVM && t != TaskType.NONE })
      yield Array[Any](x, y)).toArray
  }

  /**
   * This is a sophisticated test for standardization with the heart data set. An objective function with
   * normal input and a loss function with standardization context should produce the same result as an objective
   * function with standardized input and a plain loss function. The heart data set seems to be well-behaved, so the
   * final objective and the model coefficients can be reproduced even after many iterations.
   *
   * @param optimizerType Optimizer type
   * @param taskType Task type
   */
  @Test(dataProvider = "generateStandardizationTestData")
  def testOptimizationWithStandardization(optimizerType: OptimizerType, taskType: TaskType): Unit =
    sparkTest("testObjectivesAfterNormalization") {

      def transformVector(normalizationContext: NormalizationContext, input: Vector[Double]): Vector[Double] = {
        (normalizationContext.factors, normalizationContext.shifts) match {
          case (Some(fs), Some(ss)) =>
            require(fs.size == input.size, "Vector size and the scaling factor size are different.")
            (input - ss) :* fs
          case (Some(fs), None) =>
            require(fs.size == input.size, "Vector size and the scaling factor size are different.")
            input :* fs
          case (None, Some(ss)) =>
            require(ss.size == input.size, "Vector size and the scaling factor size are different.")
            input - ss
          case (None, None) =>
            input
        }
      }

      // Read heart data
      val heartDataRDD: RDD[LabeledPoint] = {
        val tt = getClass.getClassLoader.getResource("DriverIntegTest/input/heart.txt")
        val inputFile = tt.toString
        val rawInput = sc.textFile(inputFile, 1)
        val trainRDD: RDD[LabeledPoint] = rawInput.map(x => {
          val y = x.split(" ")
          val label = y(0).toDouble / 2 + 0.5
          val features = y.drop(1).map(z => z.split(":")(1).toDouble) :+ 1.0
          new LabeledPoint(label, DenseVector(features))
        }).persist()
        trainRDD
      }

      // Build normalization context
      val normalizationContext: NormalizationContext = {
        val dim = heartDataRDD.take(1)(0).features.size
        lazy val summary = BasicStatistics.getBasicStatistics(heartDataRDD)
        NormalizationContext(NormalizationType.STANDARDIZATION, summary, Some(dim - 1))
      }

      val normalizationBroadcast = sc.broadcast(normalizationContext)
      val noNormalizationBroadcast = sc.broadcast(NoNormalization())

      // Build the transformed rdd for validation
      val transformedRDD: RDD[LabeledPoint] = {
        val transformedRDD = heartDataRDD.map {
          case LabeledPoint(label, features, weight, offset) =>
            val transformedFeatures = transformVector(normalizationContext, features)
            new LabeledPoint(label, transformedFeatures, weight, offset)
        }.persist()
        // Verify that the transformed rdd will have the correct transformation condition
        val summaryAfterStandardization = BasicStatistics.getBasicStatistics(transformedRDD)
        summaryAfterStandardization.mean.toArray.dropRight(1).foreach(x => assertEquals(0.0, x, _delta))
        summaryAfterStandardization.variance.toArray.dropRight(1).foreach(x => assertEquals(1.0, x, _delta))
        val dim = summaryAfterStandardization.mean.size
        assertEquals(1.0, summaryAfterStandardization.mean(dim - 1), _delta)
        assertEquals(0.0, summaryAfterStandardization.variance(dim - 1), _delta)
        transformedRDD
      }

      val configuration = GLMOptimizationConfiguration()
      val objectiveFunction = taskType match {
        case TaskType.LOGISTIC_REGRESSION =>
          DistributedGLMLossFunction.create(
            configuration,
            LogisticLossFunction,
            sc,
            treeAggregateDepth = 1)

        case TaskType.LINEAR_REGRESSION =>
          DistributedGLMLossFunction.create(
            configuration,
            SquaredLossFunction,
            sc,
            treeAggregateDepth = 1)

        case TaskType.POISSON_REGRESSION =>
          DistributedGLMLossFunction.create(
            configuration,
            PoissonLossFunction,
            sc,
            treeAggregateDepth = 1)
      }
      val optimizerNorm = optimizerType match {
        case OptimizerType.LBFGS =>
          new LBFGS(tolerance = 1.0E-6, maxNumIterations = 100, normalizationContext = normalizationBroadcast)
        case OptimizerType.TRON =>
          new TRON(tolerance = 1.0E-6, maxNumIterations = 100, normalizationContext = normalizationBroadcast)
      }
      val optimizerNoNorm = optimizerType match {
        case OptimizerType.LBFGS =>
          new LBFGS(tolerance = 1.0E-6, maxNumIterations = 100, normalizationContext = noNormalizationBroadcast)
        case OptimizerType.TRON =>
          new TRON(tolerance = 1.0E-6, maxNumIterations = 100, normalizationContext = noNormalizationBroadcast)
      }

      // Train the original data with a loss function binding normalization
      val (model1, objective1) = optimizerNorm.optimize(objectiveFunction)(heartDataRDD)
      // TODO(fastier): grep for all those println (and make sure they are not logging) and clean up
      println("Optimization 1: Train the original data with a loss function binding standardization")
      println(optimizerNorm.getStateTracker.get.toString)
      println("Model 1: " + model1)
      println("Objective 1: " + objective1)
      // Train the transformed data with a normal loss function
      val (model2, objective2) = optimizerNoNorm.optimize(objectiveFunction)(transformedRDD)
      println("Optimization 2: Train the transformed data with a plain loss function")
      println(optimizerNoNorm.getStateTracker.get.toString)
      println("Model 2: " + model2)
      println("Objective 2: " + objective2)

      normalizationBroadcast.unpersist()
      noNormalizationBroadcast.unpersist()
      heartDataRDD.unpersist()
      transformedRDD.unpersist()

      // The two objective function/optimization should be exactly the same up to numerical accuracy.
      assertEquals(objective1, objective2, _delta)
      assertIterableEqualsWithTolerance(model1.toArray, model2.toArray, _delta)
    }
}
