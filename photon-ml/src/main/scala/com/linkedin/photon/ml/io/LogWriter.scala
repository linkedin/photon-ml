/*
 * Copyright 2014 LinkedIn Corp. All rights reserved.
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
package com.linkedin.photon.ml.io

import java.io.{PrintWriter, OutputStreamWriter, BufferedWriter}
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.SparkContext
import Predef.{print => sprint, println => sprintln}


/**
 * A temporary solution to write logs. Optimally this should be solved by SparkSubmit and Azkaban and we don't need to
 * keep a copy of log in the HDFS.
 *
 * @author dpeng
 */
class LogWriter(logDir: String, sc: SparkContext) {
  private val fs = FileSystem.get(sc.hadoopConfiguration)
  private val path = new Path(logDir)
  if (!fs.exists(path)) fs.mkdirs(path)

  private val logFile = new Path(logDir, "log-message.txt")
  if (fs.exists(logFile)) fs.delete(logFile, false)

  private val outputStream = fs.create(logFile)
  // The BufferedWriter writes logs.
  val log = new PrintWriter(new BufferedWriter(new OutputStreamWriter(outputStream)))

  def print(str: String): Unit = {
    log.print(str)
    sprint(str)
  }

  def println(str: String): Unit = {
    log.println(str)
    sprintln(str)
  }

  def flush(): Unit = {
    log.flush()
  }

  def close(): Unit = {
    log.close()
  }
}
