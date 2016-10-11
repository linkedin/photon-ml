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
package com.linkedin.photon.ml.data

import breeze.linalg.{SparseVector, Vector, DenseVector}
import com.linkedin.photon.ml.test.Assertions.assertIterableEqualsWithTolerance
import org.testng.Assert._
import org.testng.annotations.{DataProvider, Test}


/**
 * Test the functions in [[DataPoint]]
 */
class DataPointTest {
  val delta = 1.0E-9

  //test the class and object
  @Test
  def testApply(): Unit = {
    val features = DenseVector[Double](1.0, 10.0, 0.0, -100.0)
    val weight = 1.0
    val dataPoint = new DataPoint(features, weight)
    val expected = DataPoint(features, weight)
    assertIterableEqualsWithTolerance(dataPoint.features.toArray, expected.features.toArray, delta)
    assertEquals(dataPoint.weight, expected.weight, delta)
  }

  //test unapply()
  @Test
  def testUnapply(): Unit = {
    val features = DenseVector[Double](1.5, 13.0, -3.3, 1350.02)
    val weight = 3.2
    val dataPoint = DataPoint(features, weight)
    val featuresAndWeight = DataPoint.unapply(dataPoint)
    assertIterableEqualsWithTolerance(dataPoint.features.toArray, featuresAndWeight.get._1.toArray, delta)
    assertEquals(dataPoint.weight, featuresAndWeight.get._2, delta)
  }

  //test the extractor by case class
  @Test
  def testExtractor(): Unit = {
    val features = DenseVector[Double](2.09, 113.0, -3.3, 150.30)
    val weight = 6.4
    val dataPoint = DataPoint(features, weight)

    //test the extractor
    dataPoint match {
      case DataPoint(f, w) =>
        assertEquals(dataPoint.weight, w, delta)
        assertIterableEqualsWithTolerance(dataPoint.features.toArray, f.toArray, delta)
      case _ => throw new RuntimeException(s"extractor behavior is unexpected : [$dataPoint]")
    }
  }

  @Test(dataProvider = "dataProvider")
  def testMargin(features: Vector[Double], coef: Vector[Double], margin: Double): Unit ={
    val weight = math.random
    val dataPoint = DataPoint(features, weight)
    val actual = dataPoint.computeMargin(coef)
    assertEquals(actual, margin, delta)
  }

  @DataProvider(name = "dataProvider")
  def dataProvider(): Array[Array[Any]] = {
    Array(
      Array(DenseVector(1.0, 0.0, 0.4, 0.5), DenseVector(-1.0, -0.5, 0.1, 0.0), -0.96),
      Array(SparseVector(4)((0, 1.0), (2, 0.4), (3, 0.5)), DenseVector(-1.0, -0.5, 0.1, 0.0), -0.96),
      Array(DenseVector(1.0, 0.0, 0.4, 0.5), SparseVector(4)((0, -1.0), (1, -0.5), (2, 0.1)), -0.96),
      Array(SparseVector(4)((0, 1.0), (2, 0.4), (3, 0.5)), SparseVector(4)((0, -1.0), (1, -0.5), (2, 0.1)), -0.96)
    )
  }
}
