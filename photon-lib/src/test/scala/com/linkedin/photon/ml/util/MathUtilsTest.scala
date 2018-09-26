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

import org.testng.Assert._
import org.testng.annotations.Test

import com.linkedin.photon.ml.constants.MathConst

/**
 * Unit tests for [[MathUtils]].
 */
class MathUtilsTest {

  /**
   * Test that ln(1 + e^^x) can be computed without arithmetic overflow.
   */
  @Test
  def testLog1pExp(): Unit = {
    assertEquals(MathUtils.log1pExp(-1), 0.313261687518, MathConst.EPSILON)
    assertEquals(MathUtils.log1pExp(0), 0.69314718056, MathConst.EPSILON)
    assertEquals(MathUtils.log1pExp(1), 1.313261687518, MathConst.EPSILON)
    assertEquals(MathUtils.log1pExp(10.5), 10.500027536070, MathConst.EPSILON)
    assertEquals(MathUtils.log1pExp(100.5), 100.5, MathConst.EPSILON)
    assertEquals(MathUtils.log1pExp(10000), 10000, MathConst.EPSILON)
  }

  /**
   * Test that values less than [[MathConst.EPSILON]] are considered effectively 0.
   */
  @Test
  def testIsAlmostZero(): Unit = {
    assertTrue(MathUtils.isAlmostZero(0D))
    assertTrue(MathUtils.isAlmostZero(MathConst.EPSILON / 2D))
    assertFalse(MathUtils.isAlmostZero(MathConst.EPSILON))
    assertFalse(MathUtils.isAlmostZero(MathConst.EPSILON * 2D))
  }

  /**
   * Test that less-than comparison is correct.
   */
  @Test
  def testLessThan(): Unit = {
    assertTrue(MathUtils.lessThan(0D, 1D))
    assertFalse(MathUtils.lessThan(1D, 1D))
    assertFalse(MathUtils.lessThan(2D, 1D))
  }

  /**
   * Test that greater-than comparison is correct.
   */
  @Test
  def testGreaterThan(): Unit = {
    assertFalse(MathUtils.greaterThan(0D, 1D))
    assertFalse(MathUtils.greaterThan(1D, 1D))
    assertTrue(MathUtils.greaterThan(2D, 1D))
  }
}
