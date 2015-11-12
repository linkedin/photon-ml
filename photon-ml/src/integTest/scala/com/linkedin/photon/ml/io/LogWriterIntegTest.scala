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

import com.linkedin.photon.ml.test.SparkTestUtils
import com.linkedin.photon.ml.test.{TestTemplateWithTmpDir, SparkTestUtils}
import org.apache.spark.SparkContext
import org.testng.Assert.assertEquals
import org.testng.annotations.Test


/**
 * A simple check for [[LogWriter]]
 * @author dpeng
 */
class LogWriterIntegTest extends SparkTestUtils with TestTemplateWithTmpDir {
  private def checkContent(file: String, content: String, sc: SparkContext): Unit = {
    val actual = sc.textFile(file, 1).collect().mkString("\n")
    assertEquals(actual, content)
  }

  @Test
  def testLogWriter(): Unit = sparkTest("LogWriter") {
    val path = getTmpDir + "/" + getClass.getSimpleName
    val file = path + "/log-message.txt"
    val logger = new LogWriter(path, sc)
    logger.print("Test print")
    logger.println("Test println")
    logger.println("Test println 2")
    logger.flush()
    logger.close()
    checkContent(file, "Test printTest println\nTest println 2", sc)
  }
}
