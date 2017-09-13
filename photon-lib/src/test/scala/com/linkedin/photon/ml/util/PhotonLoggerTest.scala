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

import org.testng.Assert._
import org.testng.annotations.{DataProvider, Test}

/**
 * Tests for [[PhotonLogger]]
 */
class PhotonLoggerTest {

  @DataProvider
  def validLogLevelStrings(): Array[Array[Any]] = Array(
    Array("error", PhotonLogger.LogLevelError),
    Array("Warn", PhotonLogger.LogLevelWarn),
    Array("DeBuG", PhotonLogger.LogLevelDebug),
    Array("inFO", PhotonLogger.LogLevelInfo),
    Array("TRACE", PhotonLogger.LogLevelTrace))

  /**
   * Test that a log level name maps to the correct log level constant.
   *
   * @param logLevelString The log level name
   * @param logLevel The log level constant
   */
  @Test(dataProvider = "validLogLevelStrings")
  def testParseLogLevelString(logLevelString: String, logLevel: Int): Unit =
    assertEquals(PhotonLogger.parseLogLevelString(logLevelString), logLevel)

  @DataProvider
  def invalidLogLevelStrings(): Array[Array[Any]] = Array(
    Array("fake"),
    Array("not real"),
    Array("DeBuG_"),
    Array("inFO "))

  /**
   * Test that an invalid log level name will cause an error.
   *
   * @param logLevelString The invalid log level name
   */
  @Test(dataProvider = "invalidLogLevelStrings", expectedExceptions = Array(classOf[NoSuchElementException]))
  def testInvalidLogLevelString(logLevelString: String): Unit = PhotonLogger.parseLogLevelString(logLevelString)
}
