/*
 * Copyright 2019 LinkedIn Corp. All rights reserved.
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

import java.util.concurrent.TimeUnit

import org.mockito.ArgumentCaptor
import org.mockito.Mockito._
import org.slf4j.Logger
import org.testng.Assert._
import org.testng.annotations.Test

/**
 * Unit tests for [[Timed]].
 */
class TimedTest {

  import TimedTest._

  /**
   * Test that [[Timed.measureDuration]] will correctly execute a function and return the result while logging a message
   * before and after.
   */
  @Test
  def testMeasureDuration(): Unit = {

    val mockLogger = mock(classOf[Logger])
    val messageCaptor = ArgumentCaptor.forClass(classOf[String])

    val result = Timed.measureDuration(MESSAGE, UNITS, SUCCESS, mockLogger)

    verify(mockLogger, times(2)).info(messageCaptor.capture())

    val capturedMessages = messageCaptor.getAllValues

    assertEquals(result, SUCCESS)
    assertEquals(capturedMessages.get(0), s"$MESSAGE: begin execution")
    assertTrue(capturedMessages.get(1).startsWith(s"$MESSAGE: executed in "))
    assertTrue(capturedMessages.get(1).endsWith(s" $UNITS"))
  }

  /**
   * Test that [[Timed.apply]] will correctly execute a function and return the result while logging a message before
   * and after (measuring the duration of the function in seconds).
   */
  @Test
  def testApply(): Unit = {

    implicit val mockLogger: Logger = mock(classOf[Logger])
    val messageCaptor = ArgumentCaptor.forClass(classOf[String])

    val result = Timed(MESSAGE)(SUCCESS)

    verify(mockLogger, times(2)).info(messageCaptor.capture())

    val capturedMessages = messageCaptor.getAllValues

    assertEquals(result, SUCCESS)
    assertEquals(capturedMessages.get(0), s"$MESSAGE: begin execution")
    assertTrue(capturedMessages.get(1).startsWith(s"$MESSAGE: executed in "))
    assertTrue(capturedMessages.get(1).endsWith(s" ${TimeUnit.SECONDS}"))
  }
}

object TimedTest {

  private val UNITS = TimeUnit.MILLISECONDS
  private val MESSAGE: String = "MESSAGE"
  private val SUCCESS: String = "SUCCESS"
}
