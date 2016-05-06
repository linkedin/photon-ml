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

import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.SparkContext
import org.slf4j.helpers.{MarkerIgnoringBase, MessageFormatter}
import org.slf4j.spi.LocationAwareLogger

import java.io.{BufferedWriter, OutputStreamWriter, PrintWriter}
import java.text.SimpleDateFormat
import java.util.Date

import scala.Predef.{println => sprintln}

/**
 * Implements the SLF4J Logger interface, and writes all log messages to disk. This is currently necessary because of
 * some current log ingestion limitations on the grid, and should be removed once those limitations are addressed.
 *
 * @param logPath the location where the logfile will be stored
 * @param sc the Spark context
 */
protected[ml] class PhotonLogger(logPath: Path, sc: SparkContext) extends MarkerIgnoringBase {
  import PhotonLogger._

  def this(logPath: String, sc: SparkContext) = this(new Path(logPath), sc)

  /**
   * Default log level
   */
  var currentLogLevel = DefaultLogLevel

  /**
   * Default date formatter
   */
  var dateFormatter = new SimpleDateFormat(DefaultDateFormat)

  /**
   * Default message format
   */
  var messageFormat = DefaultMessageFormat

  /**
   * Initialize the writer
   */
  val writer = createWriter

  /**
   * Closes the log writer. The logger instance maintains an open writer, and clients are expected to call this to
   * clean up when done.
   */
  def close() {
    writer.close()
  }

  /**
   * Sets the current log level
   *
   * @param level the new log level
   */
  def setLogLevel(level: Int) {
    currentLogLevel = level
  }

  /**
   * Sets the date format for log entries
   *
   * @param format the date format, e.g. "yyyy-MM-dd"
   * @see http://docs.oracle.com/javase/6/docs/api/java/text/SimpleDateFormat.html
   */
  def setDateFormat(format: String) {
    dateFormatter = new SimpleDateFormat(DefaultDateFormat)
  }

  /**
   * Sets the log message format
   *
   * @param format the message format. Expects 3 string replacements args: formatted date/time, log level, and message
   */
  def setMessageFormat(format: String) {
    messageFormat = format
  }

  /**
   * Determines whether a particular log level is enabled
   *
   * @param level the logging level
   * @return true if the level is enabled
   */
  protected def isLevelEnabled(level: Int): Boolean = level >= currentLogLevel

  /**
   * Determines whether debug messages are enabled
   *
   * @return true if the debug level is enabled
   */
  override def isDebugEnabled(): Boolean = isLevelEnabled(LogLevelDebug)

  /**
   * Determines whether error messages are enabled
   *
   * @return true if the error level is enabled
   */
  override def isErrorEnabled(): Boolean = isLevelEnabled(LogLevelError)

  /**
   * Determines whether info messages are enabled
   *
   * @return true if the info level is enabled
   */
  override def isInfoEnabled(): Boolean = isLevelEnabled(LogLevelInfo)

  /**
   * Determines whether trace messages are enabled
   *
   * @return true if the trace level is enabled
   */
  override def isTraceEnabled(): Boolean = isLevelEnabled(LogLevelTrace)

  /**
   * Determines whether warn messages are enabled
   *
   * @return true if the warn level is enabled
   */
  override def isWarnEnabled(): Boolean = isLevelEnabled(LogLevelWarn)

  /**
   * Writes a debug level log message
   *
   * @param message the log message
   */
  override def debug(message: String): Unit = log(LogLevelDebug, message)

  /**
   * Writes a debug level log message, with a throwable stack trace
   *
   * @param message the log message
   * @param throwable the optional throwable whose stack trace should be logged
   */
  override def debug(message: String, throwable: Throwable): Unit = log(LogLevelDebug, message, Some(throwable))

  /**
   * Writes a debug level log message, with a format string and single argument
   *
   * @param format the format message
   * @param arg the format replacement argument
   * @note This is included for compatibility with the Java interface -- it's generally unnecessary in Scala, and
   * shouldn't be used.
   */
  override def debug(format: String, arg: Any): Unit = formatAndLog(LogLevelDebug, format, arg)

  /**
   * Writes a debug level log message, with a format string and two arguments
   *
   * @param format the format message
   * @param arg1 the first format replacement argument
   * @param arg2 the second format replacement argument
   * @note This is included for compatibility with the Java interface -- it's generally unnecessary in Scala, and
   * shouldn't be used.
   */
  override def debug(format: String, arg1: Any, arg2: Any): Unit = formatAndLog(LogLevelDebug, format, arg1, arg2)

  /**
   * Writes a debug level log message, with a format string and arguments
   *
   * @param format the format message
   * @param arguments the format replacement arguments
   * @note This is included for compatibility with the Java interface -- it's generally unnecessary in Scala, and
   * shouldn't be used.
   */
  override def debug(format: String, arguments: Object*): Unit = formatAndLog(LogLevelDebug, format, arguments)

  /**
   * Writes an error level log message
   *
   * @param message the log message
   */
  override def error(message: String): Unit = log(LogLevelError, message)

  /**
   * Writes an error level log message, with a throwable stack trace
   *
   * @param message the log message
   * @param throwable the optional throwable whose stack trace should be logged
   */
  override def error(message: String, throwable: Throwable): Unit = log(LogLevelError, message, Some(throwable))

  /**
   * Writes an error level log message, with a format string and single argument
   *
   * @param format the format message
   * @param arg the format replacement argument
   * @note This is included for compatibility with the Java interface -- it's generally unnecessary in Scala, and
   * shouldn't be used.
   */
  override def error(format: String, arg: Any): Unit = formatAndLog(LogLevelError, format, arg)

  /**
   * Writes an error level log message, with a format string and two arguments
   *
   * @param format the format message
   * @param arg1 the first format replacement argument
   * @param arg2 the second format replacement argument
   * @note This is included for compatibility with the Java interface -- it's generally unnecessary in Scala, and
   * shouldn't be used.
   */
  override def error(format: String, arg1: Any, arg2: Any): Unit = formatAndLog(LogLevelError, format, arg1, arg2)

  /**
   * Writes an error level log message, with a format string and arguments
   *
   * @param format the format message
   * @param arguments the format replacement arguments
   * @note This is included for compatibility with the Java interface -- it's generally unnecessary in Scala, and
   * shouldn't be used.
   */
  override def error(format: String, arguments: Object*): Unit = formatAndLog(LogLevelError, format, arguments)

  /**
   * Writes an info level log message
   *
   * @param message the log message
   */
  override def info(message: String): Unit = log(LogLevelInfo, message)

  /**
   * Writes an info level log message, with a throwable stack trace
   *
   * @param message the log message
   * @param throwable the optional throwable whose stack trace should be logged
   */
  override def info(message: String, throwable: Throwable): Unit = log(LogLevelInfo, message, Some(throwable))

  /**
   * Writes an info level log message, with a format string and single argument
   *
   * @param format the format message
   * @param arg the format replacement argument
   * @note This is included for compatibility with the Java interface -- it's generally unnecessary in Scala, and
   * shouldn't be used.
   */
  override def info(format: String, arg: Any): Unit = formatAndLog(LogLevelInfo, format, arg)

  /**
   * Writes an info level log message, with a format string and two arguments
   *
   * @param format the format message
   * @param arg1 the first format replacement argument
   * @param arg2 the second format replacement argument
   * @note This is included for compatibility with the Java interface -- it's generally unnecessary in Scala, and
   * shouldn't be used.
   */
  override def info(format: String, arg1: Any, arg2: Any): Unit = formatAndLog(LogLevelInfo, format, arg1, arg2)

  /**
   * Writes an error level log message, with a format string and arguments
   *
   * @param format the format message
   * @param arguments the format replacement arguments
   * @note This is included for compatibility with the Java interface -- it's generally unnecessary in Scala, and
   * shouldn't be used.
   */
  override def info(format: String, arguments: Object*): Unit = formatAndLog(LogLevelInfo, format, arguments)

  /**
   * Writes a trace level log message
   *
   * @param message the log message
   */
  override def trace(message: String): Unit = log(LogLevelTrace, message)

  /**
   * Writes an info level log message, with a throwable stack trace
   *
   * @param message the log message
   * @param throwable the optional throwable whose stack trace should be logged
   */
  override def trace(message: String, throwable: Throwable): Unit = log(LogLevelTrace, message, Some(throwable))

  /**
   * Writes a trace level log message, with a format string and single argument
   *
   * @param format the format message
   * @param arg the format replacement argument
   * @note This is included for compatibility with the Java interface -- it's generally unnecessary in Scala, and
   * shouldn't be used.
   */
  override def trace(format: String, arg: Any): Unit = formatAndLog(LogLevelTrace, format, arg)

  /**
   * Writes a trace level log message, with a format string and two arguments
   *
   * @param format the format message
   * @param arg1 the first format replacement argument
   * @param arg2 the second format replacement argument
   * @note This is included for compatibility with the Java interface -- it's generally unnecessary in Scala, and
   * shouldn't be used.
   */
  override def trace(format: String, arg1: Any, arg2: Any): Unit = formatAndLog(LogLevelTrace, format, arg1, arg2)

  /**
   * Writes a trace level log message, with a format string and arguments
   *
   * @param format the format message
   * @param arguments the format replacement arguments
   * @note This is included for compatibility with the Java interface -- it's generally unnecessary in Scala, and
   * shouldn't be used.
   */
  override def trace(format: String, arguments: Object*): Unit = formatAndLog(LogLevelTrace, format, arguments)

  /**
   * Writes a warn level log message
   *
   * @param message the log message
   */
  override def warn(message: String): Unit = log(LogLevelWarn, message)

  /**
   * Writes an warn level log message, with a throwable stack trace
   *
   * @param message the log message
   * @param throwable the optional throwable whose stack trace should be logged
   */
  override def warn(message: String, throwable: Throwable): Unit = log(LogLevelWarn, message, Some(throwable))

  /**
   * Writes a warn level log message, with a format string and single argument
   *
   * @param format the format message
   * @param arg the format replacement argument
   * @note This is included for compatibility with the Java interface -- it's generally unnecessary in Scala, and
   * shouldn't be used.
   */
  override def warn(format: String, arg: Any): Unit = formatAndLog(LogLevelWarn, format, arg)

  /**
   * Writes a warn level log message, with a format string and two arguments
   *
   * @param format the format message
   * @param arg1 the first format replacement argument
   * @param arg2 the second format replacement argument
   * @note This is included for compatibility with the Java interface -- it's generally unnecessary in Scala, and
   * shouldn't be used.
   */
  override def warn(format: String, arg1: Any, arg2: Any): Unit = formatAndLog(LogLevelWarn, format, arg1, arg2)

  /**
   * Writes a warn level log message, with a format string and arguments
   *
   * @param format the format message
   * @param arguments the format replacement arguments
   * @note This is included for compatibility with the Java interface -- it's generally unnecessary in Scala, and
   * shouldn't be used.
   */
  override def warn(format: String, arguments: Object*): Unit = formatAndLog(LogLevelWarn, format, arguments)

  /**
   * Formats the current date / time according to the dateFormatter
   *
   * @return formatted current date / time
   */
  protected def getFormattedDate: String = {
    this.synchronized {
      dateFormatter.format(new Date())
    }
  }

  /**
   * Returns a name for each valid log level
   *
   * @param level the log level
   * @return the level's name
   * @throws IllegalArgumentException if the log level is unrecognized
   */
  protected def getLevelName(level: Int): String = level match {
    case LogLevelDebug => "DEBUG"
    case LogLevelError => "ERROR"
    case LogLevelInfo => "INFO"
    case LogLevelTrace => "TRACE"
    case LogLevelWarn => "WARN"
    case _ =>
      throw new IllegalArgumentException(s"Unrecognized log level: $level")
  }

  /**
   * Writes the log message and optional stack trace. Messages for disabled log levels are ignored.
   *
   * @param level the log level (e.g. LogLevelInfo)
   * @param message the log message
   * @param throwable the optional throwable whose stack trace should be logged
   */
  protected def log(level: Int, message: String, throwable: Option[Throwable] = None) {
    if (!isLevelEnabled(level)) {
      return
    }

    val levelName = getLevelName(level)
    val content = String.format(messageFormat, getFormattedDate, levelName, message)

    this.synchronized {
      writer.println(content)
      sprintln(content)

      throwable match {
        case Some(t) =>
          t.printStackTrace(writer)
          t.printStackTrace(System.out)
        case None =>
      }

      writer.flush()
    }
  }

  /**
   * Applies the arguments to the format string, and then logs the result
   *
   * @param level the logging level
   * @param format the message format string
   * @param arg1 the first message format argument
   * @param arg2 the second message format argument
   */
  protected def formatAndLog(level: Int, format: String, arg1: Any, arg2: Any) {
    val formattingTuple = MessageFormatter.format(format, arg1, arg2)
    log(level, formattingTuple.getMessage)
  }

  /**
   * Applies the arguments to the format string, and then logs the result
   *
   * @param level the logging level
   * @param format the message format string
   * @param arguments the message format arguments
   */
  protected def formatAndLog(level: Int, format: String, arguments: Any*) {
    val formattingTuple = MessageFormatter.format(format, arguments)
    log(level, formattingTuple.getMessage)
  }

  /**
   * Creates a writer instance
   *
   * @return the new writer
   */
  private def createWriter: PrintWriter = {
    val fs = logPath.getFileSystem(sc.hadoopConfiguration)
    val parent = logPath.getParent

    // Create the parent directory if it doesn't exist already
    if (!fs.exists(parent)) {
      fs.mkdirs(parent)
    }

    // Overwrite any existing file
    // TODO: make this configurable?
    if (fs.exists(logPath)) {
      fs.delete(logPath, false)
    }

    new PrintWriter(
      new BufferedWriter(
        new OutputStreamWriter(
          fs.create(logPath))))
  }
}

object PhotonLogger {
  /**
   * Log level constants, inherited from SLF4J
   */
  protected[ml] val LogLevelDebug = LocationAwareLogger.DEBUG_INT
  protected[ml] val LogLevelError = LocationAwareLogger.ERROR_INT
  protected[ml] val LogLevelInfo = LocationAwareLogger.INFO_INT
  protected[ml] val LogLevelTrace = LocationAwareLogger.TRACE_INT
  protected[ml] val LogLevelWarn = LocationAwareLogger.WARN_INT

  /**
   * Default log level: INFO
   */
  protected val DefaultLogLevel = LogLevelInfo

  /**
   * Default date format
   */
  protected val DefaultDateFormat = "yyyy-MM-dd'T'HH:mm:ss.SSSZ"

  /**
   * Default log message format
   */
  protected val DefaultMessageFormat = "%s [%s] %s"
}
