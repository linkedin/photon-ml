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

import org.mockito.Mockito._
import org.testng.annotations.{AfterClass, BeforeClass, DataProvider, Test}
import org.testng.Assert._

/**
 * This class tests [[Timer]].
 */
class TimerTest {

  val now = System.nanoTime
  val duration = 17

  @Test
  def testDuration() {
    val timer = spy(new Timer)

    doReturn(now).when(timer).now
    timer.start()

    doReturn(now + duration).when(timer).now
    timer.stop()

    assertEquals(timer.duration, duration)
  }

  @Test
  def testMeasure() {
    val timer = spy(new Timer)

    doReturn(now).when(timer).now

    val result = timer.measure {
      doReturn(now + duration).when(timer).now
      "test"
    }

    assertEquals(result, ("test", duration))
  }

  @Test
  def testMark() {
    val duration = 17
    val timer = spy(new Timer)

    doReturn(now).when(timer).now
    timer.start()

    doReturn(now + duration).when(timer).now

    assertEquals(timer.mark, duration)
  }

  @Test(expectedExceptions = Array(classOf[IllegalStateException]))
  def testInvalidStart() {
    val timer = new Timer
    timer.start()
    timer.start()
  }

  @Test(expectedExceptions = Array(classOf[IllegalStateException]))
  def testInvalidStop() {
    val timer = new Timer
    timer.stop()
  }

  @Test(expectedExceptions = Array(classOf[IllegalStateException]))
  def testInvalidDuration() {
    val timer = new Timer
    timer.start()
    timer.duration
  }
}
