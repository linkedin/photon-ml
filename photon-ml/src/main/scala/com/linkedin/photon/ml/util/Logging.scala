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

import org.slf4j.{Logger, LoggerFactory}

/**
 * Logging trait for classes that write log messages.
 *
 * Note: This is a temporary solution to the backwards incompatible move in Spark 2.* to make the Logging trait
 * private. Indeed, much of this has been copied from there:
 *
 * https://github.com/apache/spark/blob/master/core/src/main/scala/org/apache/spark/internal/Logging.scala
 */
//noinspection ConvertNullInitializerToUnderscore
protected[ml] trait Logging {
  // Transient so that it can be used distributed code without incurring unnecessary serialization / io
  // We use null here because when spark deserializes an object, the transient fields are null.
  @transient private var log: Logger = null

  /**
   * Builds a unique log name for this class
   */
  protected def logName: String = {
    // Ignore trailing $'s in the class names for Scala objects
    this.getClass.getName.stripSuffix("$")
  }

  /**
   * Creates and/or returns the singleton logger instance for this class.
   */
  protected def logger: Logger = {
    if (log == null) {
      log = LoggerFactory.getLogger(logName)
    }

    log
  }
}
