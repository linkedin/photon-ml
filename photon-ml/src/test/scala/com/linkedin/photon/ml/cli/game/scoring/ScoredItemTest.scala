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
package com.linkedin.photon.ml.cli.game.scoring

import org.testng.Assert._
import org.testng.annotations.Test

class ScoredItemTest {

  @Test
  def testEquals(): Unit = {
    // Different uid, different score
    assertNotEquals(ScoredItem("0.0", 0.0), ScoredItem("1.0", 1.0))
    // Different uid, same score
    assertNotEquals(ScoredItem("0.0", 0.0), ScoredItem("1.0", 0.0))
    // Same uid, different score
    assertNotEquals(ScoredItem("1.0", 0.0), ScoredItem("1.0", 1.0))
    // Same uid, same score
    assertEquals(ScoredItem("0.0", 0.0), ScoredItem("0.0", 0.0))
  }

  @Test
  def testHashCode(): Unit = {
    // Different uid, different score
    assertNotEquals(ScoredItem("0.0", 0.0).hashCode(), ScoredItem("1.0", 1.0).hashCode())
    // Different uid, same score
    assertNotEquals(ScoredItem("0.0", 0.0).hashCode(), ScoredItem("1.0", 0.0).hashCode())
    // Same uid, different score
    assertNotEquals(ScoredItem("1.0", 0.0).hashCode(), ScoredItem("1.0", 1.0).hashCode())
    // Same uid, same score
    assertEquals(ScoredItem("0.0", 0.0).hashCode(), ScoredItem("0.0", 0.0).hashCode())
  }
}
