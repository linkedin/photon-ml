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
package com.linkedin.photon.ml.event

import org.testng.Assert._
import org.testng.annotations.Test

/**
 * Test the basic [[Event]] production/consumption cycle fully.
 */
class EventListenerTest {
  import EventListenerTest._

  @Test
  def testFullEventLoop(): Unit = {
    new EventEmitter {
      val testListener = new TestEventListener
      val testEvent = TestEvent(TEST_VALUE)

      assertEquals(testListener.getValue, TestEventListener.DEFAULT_VALUE)
      assertNotEquals(TEST_VALUE, TestEventListener.DEFAULT_VALUE)

      registerListener(testListener)
      sendEvent(testEvent)
      assertEquals(testListener.getValue, TEST_VALUE)

      clearListeners()
      assertEquals(testListener.getValue, TestEventListener.DEFAULT_VALUE)
    }
  }
}

object EventListenerTest {
  private val TEST_VALUE = 36
}
