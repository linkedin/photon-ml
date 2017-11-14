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
package com.linkedin.photon.ml.index

import org.testng.Assert._
import org.testng.annotations.Test

/**
 * Tests [[PalDBIndexMap]] implementation. This is more to guarantee that the current static serialized files
 * are still read as expected by an PalDBIndexMap instance.
 */
class PalDBIndexMapTest {
  import PalDBIndexMapTest._

  // Two tests running in parallel could also detach if the readers are thread-safe.
  @Test
  def testNoInterceptMap(): Unit = {
    val map = new PalDBIndexMap().load(
      OFFHEAP_HEART_STORE_NO_INTERCEPT,
      OFFHEAP_HEART_STORE_PARTITION_NUM.toInt,
      IndexMap.GLOBAL_NS,
      isLocal = true)

    assertEquals(map.size(), 13)

    assertEquals(map.getIndex(INTERCEPT_NAME_TERM), IndexMap.NULL_KEY)
    assertFeatureIndexMapping(map, getFeatureKey("1"), 0)
    assertFeatureIndexMapping(map, getFeatureKey("2"), 7)
    assertFeatureIndexMapping(map, getFeatureKey("3"), 5)
    assertFeatureIndexMapping(map, getFeatureKey("4"), 11)
    assertFeatureIndexMapping(map, getFeatureKey("5"), 6)
    assertFeatureIndexMapping(map, getFeatureKey("6"), 9)
    assertFeatureIndexMapping(map, getFeatureKey("7"), 3)
    assertFeatureIndexMapping(map, getFeatureKey("8"), 8)
    assertFeatureIndexMapping(map, getFeatureKey("9"), 1)
    assertFeatureIndexMapping(map, getFeatureKey("10"), 4)
    assertFeatureIndexMapping(map, getFeatureKey("11"), 12)
    assertFeatureIndexMapping(map, getFeatureKey("12"), 2)
    assertFeatureIndexMapping(map, getFeatureKey("13"), 10)
  }

  @Test
  def testWithInterceptMap(): Unit = {
    val map = new PalDBIndexMap().load(
      OFFHEAP_HEART_STORE_WITH_INTERCEPT,
      OFFHEAP_HEART_STORE_PARTITION_NUM.toInt,
      IndexMap.GLOBAL_NS,
      isLocal = true)

    assertEquals(map.getIndex(INTERCEPT_NAME_TERM), 7)
    assertFeatureIndexMapping(map, getFeatureKey("1"), 0)
    assertFeatureIndexMapping(map, getFeatureKey("2"), 8)
    assertFeatureIndexMapping(map, getFeatureKey("3"), 5)
    assertFeatureIndexMapping(map, getFeatureKey("4"), 12)
    assertFeatureIndexMapping(map, getFeatureKey("5"), 6)
    assertFeatureIndexMapping(map, getFeatureKey("6"), 10)
    assertFeatureIndexMapping(map, getFeatureKey("7"), 2)
    assertFeatureIndexMapping(map, getFeatureKey("8"), 9)
    assertFeatureIndexMapping(map, getFeatureKey("9"), 1)
    assertFeatureIndexMapping(map, getFeatureKey("10"), 4)
    assertFeatureIndexMapping(map, getFeatureKey("11"), 13)
    assertFeatureIndexMapping(map, getFeatureKey("12"), 3)
    assertFeatureIndexMapping(map, getFeatureKey("13"), 11)
  }
}

object PalDBIndexMapTest {
  private val TEST_DIR = ClassLoader.getSystemResource("PalDBIndexMapTest").getPath
  private val DELIMITER = "\u0001"
  private val INTERCEPT_NAME = "(INTERCEPT)"
  private val INTERCEPT_NAME_TERM = getFeatureKey(INTERCEPT_NAME)
  private val OFFHEAP_HEART_STORE_NO_INTERCEPT = TEST_DIR + "/paldb_offheapmap_for_heart"
  private val OFFHEAP_HEART_STORE_WITH_INTERCEPT = TEST_DIR + "/paldb_offheapmap_for_heart_with_intercept"
  private val OFFHEAP_HEART_STORE_PARTITION_NUM = "2"

  /**
   * Construct a feature key from a name and term.
   *
   * @param name The feature name
   * @param term The feature term
   * @return The unique feature key
   */
  private def getFeatureKey(name: String, term: Option[String] = None): String = name + DELIMITER + term.getOrElse("")

  /**
   * Assert that a [[PalDBIndexMap]] correctly maps between a feature name and its index.
   *
   * @param palDBIndexMap The index map
   * @param expectedName The expected feature name
   * @param expectedIdx The expected feature index
   */
  private def assertFeatureIndexMapping(palDBIndexMap: PalDBIndexMap, expectedName: String, expectedIdx: Int): Unit = {
    assertEquals(palDBIndexMap.getIndex(expectedName), expectedIdx)
    assertEquals(palDBIndexMap.getFeatureName(expectedIdx).get, expectedName)
  }
}
