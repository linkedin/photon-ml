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
package com.linkedin.photon.ml.data

import breeze.linalg.{SparseVector, Vector, DenseVector}
import com.linkedin.photon.ml.test.Assertions.assertIterableEqualsWithTolerance
import org.testng.Assert._
import org.testng.annotations.{DataProvider, Test}

/**
 * Test the functions in [[LabeledPoint]]
 *
 * @author yali
 * @author dpeng
 */
class LabeledPointTest {
  val delta = 1.0E-9

  @Test
  def testApply(): Unit = {
    val label = 1.0
    val features = DenseVector[Double](1.0, 10.0, 0.0, -100.0)
    val offset = 1.5
    val weight = 1.0
    val dataPoint = new LabeledPoint(label, features, offset, weight)
    val expected = LabeledPoint(label, features, offset, weight)
    assertEquals(dataPoint.label, expected.label, delta)
    assertIterableEqualsWithTolerance(dataPoint.features.toArray, expected.features.toArray, delta)
    assertEquals(dataPoint.offset, expected.offset, delta)
    assertEquals(dataPoint.weight, expected.weight, delta)
  }

  //test the unapply()
  @Test
  def testUnapply(): Unit = {
    val label = 1.0
    val features = DenseVector[Double](12.21, 10.0, -0.03, 10.3)
    val offset = 1.5
    val weight = 3.2
    val dataPoint = LabeledPoint(label, features, offset, weight)
    val params = LabeledPoint.unapply(dataPoint)
    assertEquals(params.get._1, label, delta)
    assertIterableEqualsWithTolerance(params.get._2.toArray, features.toArray, delta)
    assertEquals(params.get._3, offset, delta)
    assertEquals(params.get._4, weight, delta)
  }

  //test the extractor by case class
  @Test
  def testExtractor(): Unit = {
    val label = 1.0
    val features = DenseVector[Double](2.09, 113.0, -3.3, 150.30)
    val offset = 1.5
    val weight = 3.2
    val dataPoint = LabeledPoint(label, features, offset, weight)

    //test the extractor
    dataPoint match {
      case LabeledPoint(l, f, o, w) =>
        assertEquals(l, label, delta)
        assertIterableEqualsWithTolerance(f.toArray, features.toArray, delta)
        assertEquals(o, offset, delta)
        assertEquals(w, weight, delta)
      case _ => throw new RuntimeException(s"extractor behavior is unexpected : [$dataPoint]")
    }
  }

  @Test(dataProvider = "dataProvider")
  def testMargin(features: Vector[Double], offset: Double, coef: Vector[Double], margin: Double): Unit ={
    val weight = math.random
    val label = math.random
    val labeledPoint = LabeledPoint(label = label, features = features, offset = offset, weight = weight)
    val actual = labeledPoint.computeMargin(coef)
    assertEquals(actual, margin, delta)
  }

  @DataProvider(name = "dataProvider")
  def dataProvider(): Array[Array[Any]] = {
    Array(
      Array(DenseVector(1.0, 0.0, 0.4, 0.5), 1.0, DenseVector(-1.0, -0.5, 0.1, 0.0), 0.04),
      Array(SparseVector(4)((0, 1.0), (2, 0.4), (3, 0.5)), -1.0, DenseVector(-1.0, -0.5, 0.1, 0.0), -1.96),
      Array(DenseVector(1.0, 0.0, 0.4, 0.5), 0.0, SparseVector(4)((0, -1.0), (1, -0.5), (2, 0.1)), -0.96),
      Array(SparseVector(4)((0, 1.0), (2, 0.4), (3, 0.5)), -100.0, SparseVector(4)((0, -1.0), (1, -0.5), (2, 0.1)), -100.96)
    )
  }
}
