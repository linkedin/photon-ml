/*
 * Copyright 2017 LinkedIn Corp. All rights reserved.
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
package com.linkedin.photon.ml.util

import scala.math.{exp, sqrt}
import scala.util.Random

import org.testng.Assert._
import org.testng.annotations.{DataProvider, Test}

/**
 * Test for [[DoubleRange]]
 */
class DoubleRangeTest {

  /**
   * Test that an invalid range will be rejected
   */
  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testInvalidRange(): Unit = {

    val lesser = 5D
    val greater = 10D

    DoubleRange(greater, lesser)
  }

  @DataProvider
  def rangeProvider(): Array[Array[Any]] = {
    val n = 10
    val random = new Random(1L)

    Array.fill(n) {
      Array(DoubleRange(random.nextDouble, random.nextDouble + 1))
    }
  }

  /**
   * Test the transform function
   */
  @Test(dataProvider = "rangeProvider")
  def testTransform(range: DoubleRange): Unit = {
    assertEquals(range.transform(exp), DoubleRange(exp(range.start), exp(range.end)))
    assertEquals(range.transform(sqrt), DoubleRange(sqrt(range.start), sqrt(range.end)))
  }
}
