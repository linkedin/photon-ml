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

import java.nio.file.{FileSystems, Files}

import scala.io.Source

import org.testng.Assert._
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.test.{SparkTestUtils, TestTemplateWithTmpDir}

class PhotonLoggerTest extends SparkTestUtils with TestTemplateWithTmpDir {

  class TestException extends Exception

  private val LOG_REGEX_BASE = "^[0-9\\-]{10}T[0-9\\:\\.]{12}[\\-\\+][0-9]{4} \\[%s\\] %s$"
  private val DEBUG_MESSAGE = "test message 1"
  private val ERROR_MESSAGE = "test message 2"
  private val INFO_MESSAGE = "test message 3"
  private val TRACE_MESSAGE = "test message 4"
  private val WARN_MESSAGE = "test message 5"

  @Test
  def testSingleLogMessage(): Unit = sparkTest("singleLogMessage") {
    val logFile = s"$getTmpDir/singleLogMessage"
    val logger = new PhotonLogger(logFile, sc)

    try {
      logger.error(ERROR_MESSAGE)
    } finally {
      logger.close()
    }

    val fs = FileSystems.getDefault
    assertTrue(Files.exists(fs.getPath(logFile)))

    val lines = Source.fromFile(logFile).getLines().toArray
    assertEquals(lines.length, 1)
    assertTrue(lines(0).matches(LOG_REGEX_BASE.format("ERROR", ERROR_MESSAGE)))
  }

  @Test
  def testMultipleLogMessages(): Unit = sparkTest("multipleLogMessages") {
    val logFile = s"$getTmpDir/multipleLogMessages"
    val logger = new PhotonLogger(logFile, sc)

    // Trace level to catch all messages
    logger.setLogLevel(PhotonLogger.LogLevelTrace)

    try {
      logger.debug(DEBUG_MESSAGE)
      logger.error(ERROR_MESSAGE)
      logger.info(INFO_MESSAGE)
      logger.trace(TRACE_MESSAGE)
      logger.warn(WARN_MESSAGE)
    } finally {
      logger.close()
    }

    val fs = FileSystems.getDefault
    assertTrue(Files.exists(fs.getPath(logFile)))

    val lines = Source.fromFile(logFile).getLines().toArray
    assertEquals(lines.length, 5)
    assertTrue(lines(0).matches(LOG_REGEX_BASE.format("DEBUG", DEBUG_MESSAGE)))
    assertTrue(lines(1).matches(LOG_REGEX_BASE.format("ERROR", ERROR_MESSAGE)))
    assertTrue(lines(2).matches(LOG_REGEX_BASE.format("INFO", INFO_MESSAGE)))
    assertTrue(lines(3).matches(LOG_REGEX_BASE.format("TRACE", TRACE_MESSAGE)))
    assertTrue(lines(4).matches(LOG_REGEX_BASE.format("WARN", WARN_MESSAGE)))
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
  def testLogLevels(level: Int, expectedMessages: Int): Unit = sparkTest("logLevels") {
    val logFile = s"$getTmpDir/logLevels"
    val logger = new PhotonLogger(logFile, sc)

    logger.setLogLevel(level)

    try {
      logger.debug(DEBUG_MESSAGE)
      logger.error(ERROR_MESSAGE)
      logger.info(INFO_MESSAGE)
      logger.trace(TRACE_MESSAGE)
      logger.warn(WARN_MESSAGE)
    } finally {
      logger.close()
    }

    val fs = FileSystems.getDefault
    assertTrue(Files.exists(fs.getPath(logFile)))

    val lines = Source.fromFile(logFile).getLines().toArray
    assertEquals(lines.length, expectedMessages)
  }

  @Test
  def testLogMessageWithStackTrace(): Unit = sparkTest("logMessageWithStackTrace") {
    val logFile = s"$getTmpDir/multipleLogMessages"
    val logger = new PhotonLogger(logFile, sc)

    // Error level to catch only 1 message
    logger.setLogLevel(PhotonLogger.LogLevelError)

    try {
      throw new TestException
    } catch {
      case e: TestException => logger.error(ERROR_MESSAGE, e)
    } finally {
      logger.close()
    }

    val fs = FileSystems.getDefault
    assertTrue(Files.exists(fs.getPath(logFile)))

    val lines = Source.fromFile(logFile).getLines().toArray
    assertEquals(lines.length, 19) // NOTE: in IDEA, this is 34, depending on how you run the test (via gradle or not)
    assertTrue(lines(0).matches(LOG_REGEX_BASE.format("ERROR", ERROR_MESSAGE)))
    assertEquals(lines(1), "com.linkedin.photon.ml.util.PhotonLoggerTest$TestException")
  }
}
