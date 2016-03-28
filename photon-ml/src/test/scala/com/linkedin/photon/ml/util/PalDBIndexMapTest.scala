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


import com.linkedin.photon.ml.io.GLMSuite
import org.testng.annotations.Test

import org.testng.Assert._

/**
  * A class tests PalDBIndexMap implementation. This is more to gurantee that the current static serialized files
  * are still read as expected by an PalDBIndexMap instance.
  *
  *
  * also see [[com.linkedin.photon.ml.FeatureIndexingJobTest]] for more thorough tests about guaranteeing both the input
  * and the outputs are good.
  */
class PalDBIndexMapTest {
  import PalDBIndexMapTest._

  // Two tests running in parallel could also detech if the readers are thread-safe.
  @Test
  def testNoInterceptMap(): Unit = {
    val map = new PalDBIndexMap().load(OFFHEAP_HEART_STORE_NO_INTERCEPT,
        OFFHEAP_HEART_STORE_PARTITION_NUM.toInt, true)

    assertEquals(map.size(), 13)

    assertEquals(map.getIndex(GLMSuite.INTERCEPT_NAME_TERM), IndexMap.NULL_KEY)
    assertFeatureIndexMapping(map, getFeatureName("1"), 0)
    assertFeatureIndexMapping(map, getFeatureName("2"), 7)
    assertFeatureIndexMapping(map, getFeatureName("3"), 5)
    assertFeatureIndexMapping(map, getFeatureName("4"), 11)
    assertFeatureIndexMapping(map, getFeatureName("5"), 6)
    assertFeatureIndexMapping(map, getFeatureName("6"), 9)
    assertFeatureIndexMapping(map, getFeatureName("7"), 3)
    assertFeatureIndexMapping(map, getFeatureName("8"), 8)
    assertFeatureIndexMapping(map, getFeatureName("9"), 1)
    assertFeatureIndexMapping(map, getFeatureName("10"), 4)
    assertFeatureIndexMapping(map, getFeatureName("11"), 12)
    assertFeatureIndexMapping(map, getFeatureName("12"), 2)
    assertFeatureIndexMapping(map, getFeatureName("13"), 10)
  }

  @Test
  def testWithInterceptMap(): Unit = {
    val map = new PalDBIndexMap().load(OFFHEAP_HEART_STORE_WITH_INTERCEPT,
        OFFHEAP_HEART_STORE_PARTITION_NUM.toInt, true)

    assertEquals(map.getIndex(GLMSuite.INTERCEPT_NAME_TERM), 7)
    assertFeatureIndexMapping(map, getFeatureName("1"), 0)
    assertFeatureIndexMapping(map, getFeatureName("2"), 8)
    assertFeatureIndexMapping(map, getFeatureName("3"), 5)
    assertFeatureIndexMapping(map, getFeatureName("4"), 12)
    assertFeatureIndexMapping(map, getFeatureName("5"), 6)
    assertFeatureIndexMapping(map, getFeatureName("6"), 10)
    assertFeatureIndexMapping(map, getFeatureName("7"), 2)
    assertFeatureIndexMapping(map, getFeatureName("8"), 9)
    assertFeatureIndexMapping(map, getFeatureName("9"), 1)
    assertFeatureIndexMapping(map, getFeatureName("10"), 4)
    assertFeatureIndexMapping(map, getFeatureName("11"), 13)
    assertFeatureIndexMapping(map, getFeatureName("12"), 3)
    assertFeatureIndexMapping(map, getFeatureName("13"), 11)
  }
}

object PalDBIndexMapTest {
  val TEST_DIR = ClassLoader.getSystemResource("PalDBIndexMapTest").getPath
  val OFFHEAP_HEART_STORE_NO_INTERCEPT = PalDBIndexMapTest.TEST_DIR + "/paldb_offheapmap_for_heart"
  val OFFHEAP_HEART_STORE_WITH_INTERCEPT = TEST_DIR + "/paldb_offheapmap_for_heart_with_intercept"
  val OFFHEAP_HEART_STORE_PARTITION_NUM = "2"

  private def getFeatureName(name: String, term: String = null): String = {
    val termStr = if (term == null) "" else term
    name + GLMSuite.DELIMITER + termStr
  }

  private def assertFeatureIndexMapping(palDBIndexMap: PalDBIndexMap, expectedName: String, expectedIdx: Int): Unit = {
    assertEquals(palDBIndexMap.getIndex(expectedName), expectedIdx)
    assertEquals(palDBIndexMap.getFeatureName(expectedIdx), expectedName)
  }
}
