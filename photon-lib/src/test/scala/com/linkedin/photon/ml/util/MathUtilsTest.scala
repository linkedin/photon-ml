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

  @Test
  def testLog1pExp(): Unit = {
    assertEquals(MathUtils.log1pExp(-1), 0.313261687518, MathConst.EPSILON)
    assertEquals(MathUtils.log1pExp(0), 0.69314718056, MathConst.EPSILON)
    assertEquals(MathUtils.log1pExp(1), 1.313261687518, MathConst.EPSILON)
    assertEquals(MathUtils.log1pExp(10.5), 10.500027536070, MathConst.EPSILON)
    assertEquals(MathUtils.log1pExp(100.5), 100.5, MathConst.EPSILON)
    assertEquals(MathUtils.log1pExp(10000), 10000, MathConst.EPSILON)
  }

  @Test
  def testIsAlmostZero(): Unit = {
    assertTrue(MathUtils.isAlmostZero(0D))
    assertTrue(MathUtils.isAlmostZero(MathConst.EPSILON / 2D))
    assertFalse(MathUtils.isAlmostZero(MathConst.EPSILON))
    assertFalse(MathUtils.isAlmostZero(MathConst.EPSILON * 2D))
  }
}
