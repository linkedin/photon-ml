package com.linkedin.photon.ml.util

import java.io.{BufferedWriter, OutputStreamWriter, PrintWriter}

import scala.Predef.{println => sprintln}

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}

/**
 * @author xazhang
 */
class PhotonLogger(logDir: Path, configuration: Configuration) {

  def this(logDir: String, configuration: Configuration) = this(new Path(logDir), configuration)

  private val fileSystem = logDir.getFileSystem(configuration)
  private val _debug = PhotonLogger.createLogger(fileSystem, logDir, loggerName = "debug")
  private val _info = PhotonLogger.createLogger(fileSystem, logDir, loggerName = "info")
  private val _warn = PhotonLogger.createLogger(fileSystem, logDir, loggerName = "warn")
  private val _error = PhotonLogger.createLogger(fileSystem, logDir, loggerName = "error")

  def logDebug(msg: String): Unit = {
    _debug.println(msg)
    sprintln(msg)
  }

  def logInfo(msg: String): Unit = {
    _info.println(msg)
    _debug.println(msg)
    sprintln(msg)
  }

  def logWarn(msg: String): Unit = {
    _warn.println(msg)
    _info.println(msg)
    _debug.println(msg)
    sprintln(msg)
  }

  def logError(msg: String): Unit = {
    _error.println(msg)
    _warn.println(msg)
    _info.println(msg)
    _debug.println(msg)
    sprintln(msg)
  }

  def flush(): Unit = {
    _error.flush()
    _warn.flush()
    _info.flush()
    _debug.flush()
  }

  def close(): Unit = {
    _error.close()
    _warn.close()
    _info.close()
    _debug.close()
  }
}

private object PhotonLogger {
  def createLogger(fileSystem: FileSystem, logDir: Path, loggerName: String): PrintWriter = {
    val loggerPath = new Path(logDir, loggerName)
    if (fileSystem.exists(loggerPath)) fileSystem.delete(loggerPath, false)
    new PrintWriter(new BufferedWriter(new OutputStreamWriter(fileSystem.create(loggerPath))))
  }
}
