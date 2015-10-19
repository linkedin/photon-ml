package com.linkedin.photon.ml.function

import java.util.Random

import breeze.linalg.{Vector, DenseVector}
import com.linkedin.mlease.spark.data.LabeledPoint
import com.linkedin.photon.ml.data.{LabeledPoint, DataPoint}
import com.linkedin.photon.ml.function
import com.linkedin.photon.ml.test.SparkTestUtils
import org.testng.Assert._
import org.testng.annotations.{DataProvider, Test}

/**
 * Test the functions in trait [[function.DiffFunction]]
 * Only test the computeMargin() method, other derived methods are tested in ObjectiveFunctionTest
 *
 * @author yali
 */
class DiffFunctionTest extends SparkTestUtils {

  @Test
  def testComputeMargin(): Unit = {
    //create an anonymous class with the trait DiffFunction
    //test using DataPoint on some corner cases
    new {
      val _delta = 1.0E-9
    } with DiffFunction[DataPoint] {
      //override the calculateAt with a dummy implementation
      override protected[ml] def calculateAt(datum: DataPoint, coefficients: Vector[Double], cumGradient: Vector[Double]): Double = 0

      var _datum = DataPoint(DenseVector[Double](1.0, 10.0, 0.0, -100.0), 1.0)
      var _coef = DenseVector[Double](3.0, -2.0, 1.0, -0.0)
      var margin = computeMargin(_datum, _coef)
      assertEquals(margin, -17.0, _delta)

      //test 0
      _datum = DataPoint(DenseVector[Double](1.0, 10.0, 0.0, -100.0), 1.0)
      _coef = DenseVector[Double](0, 0, 0, 0)
      margin = computeMargin(_datum, _coef)
      assertEquals(margin, 0, _delta)
    }

    //test using LabeledPoint on some corner cases
    new {
      val _delta = 1.0E-9
    } with DiffFunction[LabeledPoint] {
      //override the calculateAt with a dummy implementation
      override protected[ml] def calculateAt(datum: LabeledPoint, coefficients: Vector[Double], cumGradient: Vector[Double]): Double = 0

      val _label = 1.0
      var _features = DenseVector[Double](12.21, 10.0, -0.03, 10.3)
      var _offset = 1.5
      val _weight = 3.2
      var _datum = LabeledPoint(_label, _features, _offset, _weight)
      var _coef = DenseVector[Double](1.0, 0.0, 0.0, 0.0)
      var margin = computeMargin(_datum, _coef)
      val expected = _features(0) * _coef(0) + _offset
      assertEquals(margin, expected, _delta)

      _features = DenseVector[Double](12.21, 10.0, -0.03, 10.3)
      _datum = LabeledPoint(_label, _features, _offset, _weight)
      _coef = DenseVector[Double](1.0, -2.0, 1.0, -0.0)
      margin = computeMargin(_datum, _coef)
      assertEquals(margin, -6.32, _delta)

      //test 0
      _features = DenseVector[Double](12.21, 10.0, -0.03, 10.3)
      _offset = 1.5
      _datum = LabeledPoint(_label, _features, _offset, _weight)
      _coef = DenseVector[Double](0, 0, 0, 0.0)
      margin = computeMargin(_datum, _coef)
      assertEquals(margin, _offset, _delta) //margin should equal to _offset when _coef = 0
    }
  }

  /**
   * Test the computeMargin() by comparing to the explicit form of calculation on random samples
   * @param datum Random generated samples
   */
  @Test(dataProvider = "getRandomDataPoints")
  def testComputeMarginOnRandomlyGeneratedPoints(datum: LabeledPoint): Unit =
  {
    //create an anonymous class with the trait DiffFunction
    new {

    } with DiffFunction[LabeledPoint] {
      //override the calculateAt with a dummy implementation
      override protected[ml] def calculateAt(datum: LabeledPoint, coefficients: Vector[Double], cumGradient: Vector[Double]): Double = 0

      val r: Random = new Random(DiffFunctionTest.PARAMETER_RANDOM_SEED)
      datum match {
        case LabeledPoint(_, _features, _offset, _) => {
          //init random coefficients
          val _coef: Vector[Double] = DenseVector.fill[Double] (DiffFunctionTest.PROBLEM_DIMENSION) { r.nextDouble () }
          val margin = computeMargin (datum, _coef)

          //then compute the expected margin by explicit calculation
          var _expectedMargin: Double = 0.0
          for (idx <- 0 until DiffFunctionTest.PROBLEM_DIMENSION) {
            _expectedMargin += _features(idx) * _coef(idx)
          }
          _expectedMargin += _offset
          assertEquals (margin, _expectedMargin, DiffFunctionTest.TOLERANCE, "Computed margin and expected margin don't match")
        }
        case _ => throw new IllegalArgumentException(s"Wrong data type : [$datum], expected LabeledPoint")
      }
    }
  }

  @DataProvider
  def getRandomDataPoints: Array[Array[LabeledPoint]] = {
    val randomSamples = generateBenignLocalDataSetBinaryClassification()
    randomSamples.map( {x => Array(x)} ).toArray
  }

  def generateBenignLocalDataSetBinaryClassification(): List[LabeledPoint] = {
    drawBalancedSampleFromNumericallyBenignDenseFeaturesForBinaryClassifierLocal(
      DiffFunctionTest.DATA_RANDOM_SEED,
      DiffFunctionTest.TRAINING_SAMPLES,
      DiffFunctionTest.PROBLEM_DIMENSION).map({ obj =>
      assertEquals(obj._2.length, DiffFunctionTest.PROBLEM_DIMENSION, "Samples should have expected lengths")
      new LabeledPoint(label = obj._1, obj._2, offset = 0, weight = 1)}).toList
  }
}

  object DiffFunctionTest {
    val PROBLEM_DIMENSION: Int = 10
    val TOLERANCE: Double = 1.0E-9
    val DATA_RANDOM_SEED: Int = 0
    val TRAINING_SAMPLES = PROBLEM_DIMENSION * PROBLEM_DIMENSION
    val PARAMETER_RANDOM_SEED: Int = 500
  }