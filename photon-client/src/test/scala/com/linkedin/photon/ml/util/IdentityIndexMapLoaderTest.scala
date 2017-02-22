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

import scala.collection.mutable

import org.testng.annotations.Test
import org.testng.Assert._

import com.linkedin.photon.ml.io.GLMSuite

/**
  * This class tests [[IdentityIndexMapLoader]].
  */
class IdentityIndexMapLoaderTest {
  @Test
  def testMapWithIntercept(): Unit = {
    val mapLoader = new IdentityIndexMapLoader(10, true)

    assertTrue(mapLoader.useIntercept)
    assertEquals(mapLoader.featureDimension, 10)

    val driverMap = mapLoader.indexMapForDriver()
    val rddMap = mapLoader.indexMapForRDD()
    // In unit test env, these two methods calls supposedly should return the same object
    assertTrue(driverMap == rddMap)

    assertEquals(driverMap.size, 10)

    for (i <- 0 until 9) {
      assertEquals(driverMap(String.valueOf(i)), i)
      assertEquals(driverMap.getIndex(String.valueOf(i)), i)
      assertEquals(driverMap.getFeatureName(i).get, String.valueOf(i))
    }

    assertEquals(driverMap(GLMSuite.INTERCEPT_NAME_TERM), 9)
    assertEquals(driverMap.getIndex(GLMSuite.INTERCEPT_NAME_TERM), 9)
    assertEquals(driverMap.getFeatureName(9).get, GLMSuite.INTERCEPT_NAME_TERM)

    assertTrue(driverMap.get("RANDOM_KEY").isEmpty)
    assertEquals(driverMap.getIndex("RANDOM_KEY"), IndexMap.NULL_KEY)
    assertTrue(driverMap.getFeatureName(10).isEmpty)

    val nameSet = mutable.Set[String]()
    val idxSet = mutable.Set[Int]()
    driverMap.map { case (name, idx) =>
      if (idx != 9) {
        assertEquals(name.toInt, idx)
      } else {
        assertEquals(name, GLMSuite.INTERCEPT_NAME_TERM)
      }

      nameSet.add(name)
      idxSet.add(idx)
    }

    assertEquals(nameSet.size, 10)
    assertEquals(idxSet.size, 10)
    for (i <- 0 until 10) {
      if (i != 9) {
        assertTrue(nameSet.contains(i.toString))
      } else {
        assertTrue(nameSet.contains(GLMSuite.INTERCEPT_NAME_TERM))
      }
      assertTrue(idxSet.contains(i))
    }
  }

  @Test
  def testMapWithoutIntercept(): Unit = {
    val mapLoader = new IdentityIndexMapLoader(10, false)

    assertFalse(mapLoader.useIntercept)
    assertEquals(mapLoader.featureDimension, 10)

    val driverMap = mapLoader.indexMapForDriver()
    val rddMap = mapLoader.indexMapForRDD()
    // In unit test env, these two methods calls supposedly should return the same object
    assertTrue(driverMap == rddMap)

    assertEquals(driverMap.size, 10)

    for (i <- 0 until 10) {
      assertEquals(driverMap(String.valueOf(i)), i)
      assertEquals(driverMap.getIndex(String.valueOf(i)), i)
      assertEquals(driverMap.getFeatureName(i).get, String.valueOf(i))
    }

    assertTrue(driverMap.get(GLMSuite.INTERCEPT_NAME_TERM).isEmpty)
    assertEquals(driverMap.getIndex(GLMSuite.INTERCEPT_NAME_TERM), IndexMap.NULL_KEY)
    assertEquals(driverMap.getFeatureName(9).get, "9")

    assertTrue(driverMap.get("RANDOM_KEY").isEmpty)
    assertEquals(driverMap.getIndex("RANDOM_KEY"), IndexMap.NULL_KEY)
    assertTrue(driverMap.getFeatureName(10).isEmpty)

    val nameSet = mutable.Set[String]()
    val idxSet = mutable.Set[Int]()
    driverMap.map { case (name, idx) =>
      assertEquals(name.toInt, idx)
      nameSet.add(name)
      idxSet.add(idx)
    }

    assertEquals(nameSet.size, 10)
    assertEquals(idxSet.size, 10)
    for (i <- 0 until 10) {
      assertTrue(nameSet.contains(i.toString))
      assertTrue(idxSet.contains(i))
    }
  }
}
