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

import org.testng.Assert.assertEquals
import org.testng.annotations.Test

import com.linkedin.photon.ml.test.SparkTestUtils

/**
 * Unit tests for [[BroadcastWrapper]].
 */
class BroadcastWrapperTest extends SparkTestUtils {

  @Test
  def testSimpleBroadcast(): Unit = sparkTest("testSimpleBroadcast") {
    val array = Array(1, 2, 3)
    val wrapperBroadcast: BroadcastWrapper[Array[Int]] = PhotonBroadcast(sc.broadcast(array))
    val wrapperNonBroadcast: BroadcastWrapper[Array[Int]] = PhotonNonBroadcast(array)

    assertEquals(wrapperBroadcast.value, array)
    assertEquals(wrapperNonBroadcast.value, array)
  }
}
