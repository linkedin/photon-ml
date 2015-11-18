/*
 * Copyright 2015 LinkedIn Corp. All rights reserved.
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
package com.linkedin.photon.ml

import org.testng.Assert._
import org.testng.annotations.Test

/**
 * Test DriverStage
 *
 * @author yizhou
 */
class DriverStageTest {

  @Test
  def testNames(): Unit = {
    assertEquals(DriverStage.INIT.name, "INIT")
    assertEquals(DriverStage.PREPROCESSED.name, "PREPROCESSED")
    assertEquals(DriverStage.TRAINED.name, "TRAINED")
    assertEquals(DriverStage.VALIDATED.name, "VALIDATED")
  }

  @Test
  def testOrderNum(): Unit = {
    assertEquals(DriverStage.INIT.order, 0)
    assertEquals(DriverStage.PREPROCESSED.order, 1)
    assertEquals(DriverStage.TRAINED.order, 2)
    assertEquals(DriverStage.VALIDATED.order, 3)
  }

  @Test
  def testOrder(): Unit = {
    assertTrue(DriverStage.INIT == DriverStage.INIT)
    assertTrue(DriverStage.PREPROCESSED == DriverStage.PREPROCESSED)
    assertTrue(DriverStage.TRAINED == DriverStage.TRAINED)
    assertTrue(DriverStage.VALIDATED == DriverStage.VALIDATED)

    assertTrue(DriverStage.INIT < DriverStage.VALIDATED)
    assertTrue(DriverStage.INIT <= DriverStage.VALIDATED)
    assertTrue(DriverStage.TRAINED > DriverStage.PREPROCESSED)
    assertTrue(DriverStage.TRAINED >= DriverStage.PREPROCESSED)
  }

  @Test
  def testSort(): Unit = {
    val stages1 = Array(DriverStage.TRAINED, DriverStage.VALIDATED, DriverStage.PREPROCESSED, DriverStage.INIT)
    assertEquals(stages1.sortWith(_ < _), Array(DriverStage.INIT, DriverStage.PREPROCESSED, DriverStage.TRAINED,
        DriverStage.VALIDATED))

    val stages2 = Array(DriverStage.INIT, DriverStage.PREPROCESSED, DriverStage.TRAINED)
    assertEquals(stages2.sortWith(_ < _), Array(DriverStage.INIT, DriverStage.PREPROCESSED, DriverStage.TRAINED))
  }
}
