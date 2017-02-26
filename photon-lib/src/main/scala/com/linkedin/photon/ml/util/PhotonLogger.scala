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

import java.io.{BufferedWriter, PrintWriter, _}
import java.text.SimpleDateFormat
import java.util.{Date, UUID}

import scala.Predef.{println => sprintln}

import org.apache.commons.io.FileUtils
import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.slf4j.helpers.{MarkerIgnoringBase, MessageFormatter}
import org.slf4j.spi.LocationAwareLogger

/**
 * Implements the SLF4J Logger interface, and writes all log messages to disk. This is currently necessary because of
 * some current log ingestion limitations on the grid, and should be removed once those limitations are addressed.
 *
 * @param logPath The location where the logfile will be stored
 * @param sc The Spark context
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
   * A local tmp file path for the logger
   */
  val tmpLocalPath: Path = new Path(FileUtils.getTempDirectoryPath, UUID.randomUUID().toString)

  /**
   * Initialize the writer
   */
  val writer = createWriter

  /**
   * Closes the log writer. The logger instance maintains an open writer, and clients are expected to call this to
   * clean up when done.
   */
  def close(): Unit = {
    try {
      val fs = logPath.getFileSystem(sc.hadoopConfiguration)
      writer.close()

      val parent = logPath.getParent
      // Create the parent directory if it doesn't exist already
      if (!fs.exists(parent)) {
        fs.mkdirs(parent)
      }
      fs.copyFromLocalFile(tmpLocalPath, logPath)
    } catch {
      case e: Exception => throw e
    } finally {
      new File(tmpLocalPath.toString).delete()
    }
  }

  /**
   * Sets the current log level.
   *
   * @param level The new log level
   */
  def setLogLevel(level: Int): Unit = {
    currentLogLevel = level
  }

  /**
   * Sets the date format for log entries.
   *
   * @param format the date format, e.g. "yyyy-MM-dd"
   * @see http://docs.oracle.com/javase/6/docs/api/java/text/SimpleDateFormat.html
   */
  def setDateFormat(format: String): Unit = {
    dateFormatter = new SimpleDateFormat(DefaultDateFormat)
  }

  /**
   * Sets the log message format.
   *
   * @param format The message format. Expects 3 string replacements args: formatted date/time, log level, and message
   */
  def setMessageFormat(format: String): Unit = {
    messageFormat = format
  }

  /**
   * Determines whether a particular log level is enabled.
   *
   * @param level The logging level
   * @return True if the level is enabled
   */
  protected def isLevelEnabled(level: Int): Boolean = level >= currentLogLevel

  /**
   * Determines whether debug messages are enabled.
   *
   * @return True if the debug level is enabled
   */
  override def isDebugEnabled: Boolean = isLevelEnabled(LogLevelDebug)

  /**
   * Determines whether error messages are enabled.
   *
   * @return True if the error level is enabled
   */
  override def isErrorEnabled: Boolean = isLevelEnabled(LogLevelError)

  /**
   * Determines whether info messages are enabled.
   *
   * @return True if the info level is enabled
   */
  override def isInfoEnabled: Boolean = isLevelEnabled(LogLevelInfo)

  /**
   * Determines whether trace messages are enabled.
   *
   * @return True if the trace level is enabled
   */
  override def isTraceEnabled: Boolean = isLevelEnabled(LogLevelTrace)

  /**
   * Determines whether warn messages are enabled.
   *
   * @return True if the warn level is enabled
   */
  override def isWarnEnabled: Boolean = isLevelEnabled(LogLevelWarn)

  /**
   * Writes a debug level log message.
   *
   * @param message The log message
   */
  override def debug(message: String): Unit = log(LogLevelDebug, message)

  /**
   * Writes a debug level log message, with a throwable stack trace.
   *
   * @param message The log message
   * @param throwable The optional throwable whose stack trace should be logged
   */
  override def debug(message: String, throwable: Throwable): Unit = log(LogLevelDebug, message, Some(throwable))

  /**
   * Writes a debug level log message, with a format string and single argument.
   *
   * @note This is included for compatibility with the Java interface -- it's generally unnecessary in Scala, and
   *       shouldn't be used.
   *
   * @param format The format message
   * @param arg The format replacement argument
   */
  override def debug(format: String, arg: Any): Unit = formatAndLog(LogLevelDebug, format, arg)

  /**
   * Writes a debug level log message, with a format string and two arguments.
   *
   * @note This is included for compatibility with the Java interface -- it's generally unnecessary in Scala, and
   * shouldn't be used.
   *
   * @param format The format message
   * @param arg1 The first format replacement argument
   * @param arg2 The second format replacement argument
   */
  override def debug(format: String, arg1: Any, arg2: Any): Unit = formatAndLog(LogLevelDebug, format, arg1, arg2)

  /**
   * Writes a debug level log message, with a format string and arguments.
   *
   * @note This is included for compatibility with the Java interface -- it's generally unnecessary in Scala, and
   * shouldn't be used.
   *
   * @param format The format message
   * @param arguments The format replacement arguments
   */
  override def debug(format: String, arguments: Object*): Unit = formatAndLog(LogLevelDebug, format, arguments)

  /**
   * Writes an error level log message.
   *
   * @param message The log message
   */
  override def error(message: String): Unit = log(LogLevelError, message)

  /**
   * Writes an error level log message, with a throwable stack trace.
   *
   * @param message The log message
   * @param throwable The optional throwable whose stack trace should be logged
   */
  override def error(message: String, throwable: Throwable): Unit = log(LogLevelError, message, Some(throwable))

  /**
   * Writes an error level log message, with a format string and single argument.
   *
   * @note This is included for compatibility with the Java interface -- it's generally unnecessary in Scala, and
   * shouldn't be used.
   *
   * @param format The format message
   * @param arg The format replacement argument
   */
  override def error(format: String, arg: Any): Unit = formatAndLog(LogLevelError, format, arg)

  /**
   * Writes an error level log message, with a format string and two arguments.
   *
   * @note This is included for compatibility with the Java interface -- it's generally unnecessary in Scala, and
   * shouldn't be used.
   *
   * @param format The format message
   * @param arg1 The first format replacement argument
   * @param arg2 The second format replacement argument
   */
  override def error(format: String, arg1: Any, arg2: Any): Unit = formatAndLog(LogLevelError, format, arg1, arg2)

  /**
   * Writes an error level log message, with a format string and arguments.
   *
   * @note This is included for compatibility with the Java interface -- it's generally unnecessary in Scala, and
   * shouldn't be used.
   *
   * @param format The format message
   * @param arguments The format replacement arguments
   */
  override def error(format: String, arguments: Object*): Unit = formatAndLog(LogLevelError, format, arguments)

  /**
   * Writes an info level log message.
   *
   * @param message The log message
   */
  override def info(message: String): Unit = log(LogLevelInfo, message)

  /**
   * Writes an info level log message, with a throwable stack trace.
   *
   * @param message The log message
   * @param throwable The optional throwable whose stack trace should be logged
   */
  override def info(message: String, throwable: Throwable): Unit = log(LogLevelInfo, message, Some(throwable))

  /**
   * Writes an info level log message, with a format string and single argument.
   *
   * @note This is included for compatibility with the Java interface -- it's generally unnecessary in Scala, and
   * shouldn't be used.
   *
   * @param format The format message
   * @param arg The format replacement argument
   */
  override def info(format: String, arg: Any): Unit = formatAndLog(LogLevelInfo, format, arg)

  /**
   * Writes an info level log message, with a format string and two arguments.
   *
   * @note This is included for compatibility with the Java interface -- it's generally unnecessary in Scala, and
   * shouldn't be used.
   *
   * @param format The format message
   * @param arg1 The first format replacement argument
   * @param arg2 The second format replacement argument
   */
  override def info(format: String, arg1: Any, arg2: Any): Unit = formatAndLog(LogLevelInfo, format, arg1, arg2)

  /**
   * Writes an error level log message, with a format string and arguments.
   *
   * @note This is included for compatibility with the Java interface -- it's generally unnecessary in Scala, and
   * shouldn't be used.
   *
   * @param format The format message
   * @param arguments The format replacement arguments
   */
  override def info(format: String, arguments: Object*): Unit = formatAndLog(LogLevelInfo, format, arguments)

  /**
   * Writes a trace level log message.
   *
   * @param message The log message
   */
  override def trace(message: String): Unit = log(LogLevelTrace, message)

  /**
   * Writes an info level log message, with a throwable stack trace.
   *
   * @param message The log message
   * @param throwable The optional throwable whose stack trace should be logged
   */
  override def trace(message: String, throwable: Throwable): Unit = log(LogLevelTrace, message, Some(throwable))

  /**
   * Writes a trace level log message, with a format string and single argument.
   *
   * @note This is included for compatibility with the Java interface -- it's generally unnecessary in Scala, and
   * shouldn't be used.
   *
   * @param format The format message
   * @param arg The format replacement argument
   */
  override def trace(format: String, arg: Any): Unit = formatAndLog(LogLevelTrace, format, arg)

  /**
   * Writes a trace level log message, with a format string and two arguments.
   *
   * @note This is included for compatibility with the Java interface -- it's generally unnecessary in Scala, and
   * shouldn't be used.
   *
   * @param format The format message
   * @param arg1 The first format replacement argument
   * @param arg2 The second format replacement argument
   */
  override def trace(format: String, arg1: Any, arg2: Any): Unit = formatAndLog(LogLevelTrace, format, arg1, arg2)

  /**
   * Writes a trace level log message, with a format string and arguments.
   *
   * @note This is included for compatibility with the Java interface -- it's generally unnecessary in Scala, and
   * shouldn't be used.
   *
   * @param format The format message
   * @param arguments The format replacement arguments
   */
  override def trace(format: String, arguments: Object*): Unit = formatAndLog(LogLevelTrace, format, arguments)

  /**
   * Writes a warn level log message.
   *
   * @param message The log message
   */
  override def warn(message: String): Unit = log(LogLevelWarn, message)

  /**
   * Writes an warn level log message, with a throwable stack trace.
   *
   * @param message The log message
   * @param throwable The optional throwable whose stack trace should be logged
   */
  override def warn(message: String, throwable: Throwable): Unit = log(LogLevelWarn, message, Some(throwable))

  /**
   * Writes a warn level log message, with a format string and single argument.
   *
   * @note This is included for compatibility with the Java interface -- it's generally unnecessary in Scala, and
   * shouldn't be used.
   *
   * @param format The format message
   * @param arg The format replacement argument
   */
  override def warn(format: String, arg: Any): Unit = formatAndLog(LogLevelWarn, format, arg)

  /**
   * Writes a warn level log message, with a format string and two arguments.
   *
   * @note This is included for compatibility with the Java interface -- it's generally unnecessary in Scala, and
   * shouldn't be used.
   *
   * @param format The format message
   * @param arg1 The first format replacement argument
   * @param arg2 The second format replacement argument
   */
  override def warn(format: String, arg1: Any, arg2: Any): Unit = formatAndLog(LogLevelWarn, format, arg1, arg2)

  /**
   * Writes a warn level log message, with a format string and arguments.
   *
   * @note This is included for compatibility with the Java interface -- it's generally unnecessary in Scala, and
   * shouldn't be used.
   *
   * @param format The format message
   * @param arguments The format replacement arguments
   */
  override def warn(format: String, arguments: Object*): Unit = formatAndLog(LogLevelWarn, format, arguments)

  /**
   * Formats the current date / time according to the dateFormatter.
   *
   * @return Formatted current date / time
   */
  protected def getFormattedDate: String = {
    this.synchronized {
      dateFormatter.format(new Date())
    }
  }

  /**
   * Returns a name for each valid log level.
   *
   * @param level The log level
   * @throws IllegalArgumentException if the log level is unrecognized
   * @return The level's name
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
   * @param level The log level (e.g. LogLevelInfo)
   * @param message The log message
   * @param throwable The optional throwable whose stack trace should be logged
   */
  protected def log(level: Int, message: String, throwable: Option[Throwable] = None): Unit = {
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
   * Applies the arguments to the format string, and then logs the result.
   *
   * @param level The logging level
   * @param format The message format string
   * @param arg1 The first message format argument
   * @param arg2 The second message format argument
   */
  protected def formatAndLog(level: Int, format: String, arg1: Any, arg2: Any): Unit = {
    val formattingTuple = MessageFormatter.format(format, arg1, arg2)
    log(level, formattingTuple.getMessage)
  }

  /**
   * Applies the arguments to the format string, and then logs the result.
   *
   * @param level The logging level
   * @param format The message format string
   * @param arguments The message format arguments
   */
  protected def formatAndLog(level: Int, format: String, arguments: Any*): Unit = {
    val formattingTuple = MessageFormatter.format(format, arguments)
    log(level, formattingTuple.getMessage)
  }

  /**
   * Creates a writer instance.
   *
   * @return The new writer
   */
  private def createWriter: PrintWriter = {
    new PrintWriter(new BufferedWriter(new FileWriter(new File(tmpLocalPath.toString))))
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
