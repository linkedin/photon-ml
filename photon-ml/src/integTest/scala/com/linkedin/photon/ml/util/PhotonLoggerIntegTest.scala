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

import org.testng.annotations.{DataProvider, Test}
import org.testng.Assert._

import java.nio.file.{Files, FileSystems, Path}
import scala.io.Source

import com.linkedin.photon.ml.test.{CommonTestUtils, SparkTestUtils, TestTemplateWithTmpDir}

class PhotonLoggerIntegTest extends SparkTestUtils with TestTemplateWithTmpDir {

  class TestException extends Exception

  @Test
  def testSingleLogMessage = sparkTest("singleLogMessage") {
    val logFile = s"$getTmpDir/singleLogMessage"
    val logger = new PhotonLogger(logFile, sc)

    try {
      logger.error("test message")

    } finally {
      logger.close
    }

    val fs = FileSystems.getDefault()
    assertTrue(Files.exists(fs.getPath(logFile)))

    val lines = Source.fromFile(logFile).getLines.toArray
    assertEquals(lines.size, 1)
    assertTrue(lines(0).matches("^[0-9T\\-\\:\\.]* \\[ERROR\\] test message$"))
  }

  @Test
  def testMultipleLogMessages = sparkTest("multipleLogMessages") {
    val logFile = s"$getTmpDir/multipleLogMessages"
    val logger = new PhotonLogger(logFile, sc)

    // Trace level to catch all messages
    logger.setLogLevel(PhotonLogger.LogLevelTrace)

    try {
      logger.debug("test message 1")
      logger.error("test message 2")
      logger.info("test message 3")
      logger.trace("test message 4")
      logger.warn("test message 5")

    } finally {
      logger.close
    }

    val fs = FileSystems.getDefault()
    assertTrue(Files.exists(fs.getPath(logFile)))

    val lines = Source.fromFile(logFile).getLines.toArray
    assertEquals(lines.size, 5)
    assertTrue(lines(0).matches("^[0-9T\\-\\:\\.]* \\[DEBUG\\] test message 1$"))
    assertTrue(lines(1).matches("^[0-9T\\-\\:\\.]* \\[ERROR\\] test message 2$"))
    assertTrue(lines(2).matches("^[0-9T\\-\\:\\.]* \\[INFO\\] test message 3$"))
    assertTrue(lines(3).matches("^[0-9T\\-\\:\\.]* \\[TRACE\\] test message 4$"))
    assertTrue(lines(4).matches("^[0-9T\\-\\:\\.]* \\[WARN\\] test message 5$"))
  }

  @DataProvider
  def logLevelTestDataProvider: Array[Array[Any]] = {
    Array(
      Array(PhotonLogger.LogLevelTrace, 5),
      Array(PhotonLogger.LogLevelDebug, 4),
      Array(PhotonLogger.LogLevelInfo, 3),
      Array(PhotonLogger.LogLevelWarn, 2),
      Array(PhotonLogger.LogLevelError, 1))
  }

  @Test(dataProvider = "logLevelTestDataProvider")
  def testLogLevels(level: Int, expectedMessages: Int) = sparkTest("logLevels") {
    val logFile = s"$getTmpDir/logLevels"
    val logger = new PhotonLogger(logFile, sc)

    logger.setLogLevel(level)

    try {
      logger.debug("test message 1")
      logger.error("test message 2")
      logger.info("test message 3")
      logger.trace("test message 4")
      logger.warn("test message 5")

    } finally {
      logger.close
    }

    val fs = FileSystems.getDefault()
    assertTrue(Files.exists(fs.getPath(logFile)))

    val lines = Source.fromFile(logFile).getLines.toArray
    assertEquals(lines.size, expectedMessages)
  }

  @Test
  def testLogMessageWithStackTrace = sparkTest("logMessageWithStackTrace") {
    val logFile = s"$getTmpDir/multipleLogMessages"
    val logger = new PhotonLogger(logFile, sc)

    // Error level to catch only 1 message
    logger.setLogLevel(PhotonLogger.LogLevelError)

    try {
        throw new TestException

    } catch {
      case e: TestException =>
        logger.error("test message 2", e)

    } finally {
      logger.close
    }

    val fs = FileSystems.getDefault()
    assertTrue(Files.exists(fs.getPath(logFile)))

    val lines = Source.fromFile(logFile).getLines.toArray
    assertEquals(lines.size, 19)
    assertTrue(lines(0).matches("^[0-9T\\-\\:\\.]* \\[ERROR\\] test message 2$"))
    assertEquals(lines(1), "com.linkedin.photon.ml.util.PhotonLoggerIntegTest$TestException")
  }
}
