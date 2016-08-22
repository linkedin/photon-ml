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
package com.linkedin.photon.ml.util

import org.testng.annotations.Test
import org.testng.Assert._

/**
 * Tests the [[DefaultIndexMap]] implementation
 */
class DefaultIndexMapTest {

  val indexMap = new DefaultIndexMap(Map("a" -> 0, "b" -> 1))

  @Test
  def testForwardLookup(): Unit = {
    assertEquals(indexMap.getIndex("a"), 0)
    assertEquals(indexMap.getIndex("b"), 1)
  }

  @Test
  def testReverseLookup(): Unit = {
    assertEquals(indexMap.getFeatureName(0), Some("a"))
    assertEquals(indexMap.getFeatureName(1), Some("b"))
  }

  @Test
  def testSize(): Unit = {
    assertEquals(indexMap.size, 2)
    assertEquals(new DefaultIndexMap(Map("a" -> 0)).size, 1)
    assertEquals(new DefaultIndexMap(Map("a" -> 0, "b" -> 1, "c" -> 2)).size, 3)
    assertEquals(new DefaultIndexMap("abcdefghijklmnopqrstuvwxyz".split("").zipWithIndex.toMap).size, 26)
  }

  @Test
  def testEmpty(): Unit = {
    assertEquals(indexMap.isEmpty, false)
    assertEquals(new DefaultIndexMap(Map.empty[String, Int]).isEmpty, true)
  }
}
