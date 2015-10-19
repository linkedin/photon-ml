package com.linkedin.photon.ml.function

import breeze.linalg.DenseVector
import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.test.Assertions
import Assertions._
import org.testng.Assert._
import org.testng.annotations.Test

/**
 * Test the functions in [[LogisticLossFunction]]
 * More tests by numerical methods please see com.linkedin.photon.ml.function.ObjectiveFunctionTest
 * @author yali
 */
class LogisticLossFunctionTest {

  //test calculateAt() on some corner cases
  @Test
  def testCalculateAt(): Unit = {
    val _delta = 1.0E-9
    //test label 1.0
    var _label = 1.0
    val _features = DenseVector[Double](12.21, 10.0, -0.03, 10.3)
    val _offset = 1.5
    val _weight = 1.0
    var _datum = LabeledPoint(_label, _features, _offset, _weight)
    val _coff = DenseVector[Double](1.0, 12.3, -21.0, 0.0)
    val _cumGradient = DenseVector[Double](0, 0, 0, 0)

    val logisticLossFunc = new LogisticLossFunction()
    var value = logisticLossFunc.calculateAt(_datum, _coff, _cumGradient)
    //compute the expected value by explicit computation
    val margin = _features.dot(_coff) + _offset
    var expected = _label * math.log(1 + math.exp(-margin)) + (1 - _label) * math.log(1 + math.exp(margin))
    assertEquals(value, expected, _delta)

    //test label 0.0
    _label = 0.0
    _datum = LabeledPoint(_label, _features, _offset, _weight)
    value = logisticLossFunc.calculateAt(_datum, _coff, _cumGradient)
    expected = _label * math.log(1 + math.exp(-margin)) + (1 - _label) * math.log(1 + math.exp(margin))
    assertEquals(value, expected, _delta)
  }

  //test gradientAt() on some corner cases
  @Test
  def testGradientAt(): Unit = {
    val _delta = 1.0E-9
    //test label 1.0
    var _label = 1.0
    val _features = DenseVector[Double](12.21, 10.0, -0.03, 10.3)
    val _offset = 0
    val _weight = 1.0
    var _datum = LabeledPoint(_label, _features, _offset, _weight)
    val _coff = DenseVector[Double](1.0, 1.0, 1.0, 1.0)

    val logisticLossFunc = new LogisticLossFunction()
    val gradient = logisticLossFunc.gradientAt(_datum, _coff)

    //calculate it explicitly
    val margin = _features.dot(_coff) + _offset
    val expected = _features * (1.0 / (1.0 + math.exp(-margin)) - 1.0)
    assertIterableEqualsWithTolerance(gradient.toArray, expected.toArray, _delta)
  }

  //test hessianVectorAt() on corner cases
  @Test
  def testHessianVectorAt(): Unit = {
  //test 0
    val _delta = 1.0E-9
    val _label = 1.0
    var _features = DenseVector[Double](0.0, 0.0, 0.0, 0.0)
    val _offset = 0
    val _weight = 1.0
    var _vector = DenseVector[Double](0.0, 0.0, 0.0, 0.0)
    var _datum = LabeledPoint(_label, _features, _offset, _weight)
    var _coef = DenseVector[Double](0.0, 0.0, 0.0, 0.0)
    var _cumHessianVector = DenseVector[Double](0.0, 0.0, 0.0, 0.0)
    val logisticLossFunc = new LogisticLossFunction()
    logisticLossFunc.hessianVectorAt(_datum, _coef, _vector, _cumHessianVector)
    var expected = DenseVector[Double](0.0, 0.0, 0.0, 0.0)
    assertIterableEqualsWithTolerance(_cumHessianVector.toArray, expected.toArray, _delta)

    //another normal case
    _features = DenseVector[Double](1.0, 0.0, 0.0, 0.0)
    _vector = DenseVector[Double](1.0, 0.0, 0.0, 0.0)
    _coef = DenseVector[Double](1.0, 0.0, 0.0, 0.0)
    _datum = LabeledPoint(_label, _features, _offset, _weight)
    _cumHessianVector = DenseVector[Double](0.0, 0.0, 0.0, 0.0)
    logisticLossFunc.hessianVectorAt(_datum, _coef, _vector, _cumHessianVector)

    val margin = _features.dot(_coef)
    val sigma = 1.0 / (1.0 + math.exp(-margin))
    val D = sigma * (1-sigma)
    expected = DenseVector[Double](_weight * D * margin, 0.0, 0.0, 0.0)
    assertIterableEqualsWithTolerance(_cumHessianVector.toArray, expected.toArray, _delta)
  }
}