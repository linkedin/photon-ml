/*
 * Copyright 2014 LinkedIn Corp. All rights reserved.
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

import breeze.linalg.DenseVector

import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.test.Assertions
import Assertions._

import org.testng.Assert._
import org.testng.annotations.Test

/**
 * Test some edge cases of the functions in [[com.linkedin.photon.ml.function.LogisticLossFunction]]
 * More tests by numerical methods please see [[com.linkedin.photon.ml.function.ObjectiveFunctionTest]]
 * @author yali
 * @author dpeng
 */
class LogisticLossFunctionTest {

  @Test
  def testCalculate(): Unit = {
    val _delta = 1.0E-9
    //test label 1.0
    var _label = 1.0
    val _features = DenseVector[Double](12.21, 10.0, -0.03, 10.3)
    val _offset = 1.5
    val _weight = 1.0
    var _datum = LabeledPoint(_label, _features, _offset, _weight)
    val _coff = DenseVector[Double](1.0, 12.3, -21.0, 0.0)

    val logisticLossFunc = new LogisticLossFunction()
    val (value1, _) = logisticLossFunc.calculate(Seq(_datum), _coff)
    //compute the expected value by explicit computation
    val margin = _features.dot(_coff) + _offset
    var expected = _label * math.log(1 + math.exp(-margin)) + (1 - _label) * math.log(1 + math.exp(margin))
    assertEquals(value1, expected, _delta)

    //test label 0.0
    _label = 0.0
    _datum = LabeledPoint(_label, _features, _offset, _weight)
    val (value2, _) = logisticLossFunc.calculate(Seq(_datum), _coff)
    expected = _label * math.log(1 + math.exp(-margin)) + (1 - _label) * math.log(1 + math.exp(margin))
    assertEquals(value2, expected, _delta)
  }

  @Test
  def testGradient(): Unit = {
    val _delta = 1.0E-9
    //test label 1.0
    var _label = 1.0
    val _features = DenseVector[Double](12.21, 10.0, -0.03, 10.3)
    val _offset = 0
    val _weight = 1.0
    var _datum = LabeledPoint(_label, _features, _offset, _weight)
    val _coff = DenseVector[Double](1.0, 1.0, 1.0, 1.0)

    val logisticLossFunc = new LogisticLossFunction()
    val (_, gradient) = logisticLossFunc.calculate(Seq(_datum), _coff)

    //calculate it explicitly
    val margin = _features.dot(_coff) + _offset
    val expected = _features * (1.0 / (1.0 + math.exp(-margin)) - 1.0)
    assertIterableEqualsWithTolerance(gradient.toArray, expected.toArray, _delta)
  }

  @Test
  def testHessianVector(): Unit = {
  //test 0
    val _delta = 1.0E-9
    val _label = 1.0
    var _features = DenseVector[Double](0.0, 0.0, 0.0, 0.0)
    val _offset = 0
    val _weight = 1.0
    var _vector = DenseVector[Double](0.0, 0.0, 0.0, 0.0)
    var _datum = LabeledPoint(_label, _features, _offset, _weight)
    var _coef = DenseVector[Double](0.0, 0.0, 0.0, 0.0)
    val logisticLossFunc = new LogisticLossFunction()
    val result1 = logisticLossFunc.hessianVector(Seq(_datum), _coef, _vector)
    var expected = DenseVector[Double](0.0, 0.0, 0.0, 0.0)
    assertIterableEqualsWithTolerance(result1.toArray, expected.toArray, _delta)

    //another normal case
    _features = DenseVector[Double](1.0, 0.0, 0.0, 0.0)
    _vector = DenseVector[Double](1.0, 0.0, 0.0, 0.0)
    _coef = DenseVector[Double](1.0, 0.0, 0.0, 0.0)
    _datum = LabeledPoint(_label, _features, _offset, _weight)
    val result2 = logisticLossFunc.hessianVector(Seq(_datum), _coef, _vector)

    val margin = _features.dot(_coef)
    val sigma = 1.0 / (1.0 + math.exp(-margin))
    val D = sigma * (1-sigma)
    expected = DenseVector[Double](_weight * D * margin, 0.0, 0.0, 0.0)
    assertIterableEqualsWithTolerance(result2.toArray, expected.toArray, _delta)
  }
}
