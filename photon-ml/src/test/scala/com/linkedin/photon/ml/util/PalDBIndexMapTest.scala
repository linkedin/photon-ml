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
  * Created by yizhou on 3/23/16.
  */
class PalDBIndexMapTest {

  @Test
  def testMap(): Unit = {
    val map = new PalDBIndexMap().load("/tmp/index-output", 2)

    assertEquals(map.size(), 13)

    assertEquals(map.getIndex(new java.lang.String("2" + "\u0001")), 0)
    assertEquals(map.getIndex(new java.lang.String("8" + "\u0001")), 1)
    assertEquals(map.getIndex("5" + "\u0001"), 3)
    assertEquals(map.getIndex("9" + "\u0001"), 4)
  }
}
