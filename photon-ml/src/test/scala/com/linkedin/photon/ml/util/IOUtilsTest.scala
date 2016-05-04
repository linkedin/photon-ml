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

import com.linkedin.photon.ml.test.TestTemplateWithTmpDir
import org.apache.hadoop.conf.Configuration
import org.testng.Assert
import org.testng.annotations.{DataProvider, Test}

import scala.collection.mutable.ArrayBuffer


/**
 * This class tests [[IOUtils]] can correctly read and write primitive/ASCII strings and
 * international/UTF-8 strings.
 */
class IOUtilsTest extends TestTemplateWithTmpDir {
  @Test(dataProvider = "dataProvider")
  def testReadAndWrite(dir: String, testString: String): Unit = {
    val tmpDir = getTmpDir + "/" + dir
    val conf = new Configuration()
    IOUtils.writeStringsToHDFS(List(testString).iterator, tmpDir, conf, true)
    val strings = IOUtils.readStringsFromHDFS(tmpDir, conf)
    Assert.assertEquals(strings, ArrayBuffer(testString))
  }

  @DataProvider
  def dataProvider(): Array[Array[Any]] = {
    Array(
      Array("ASCII", "Test string"),
      Array("UTF-8", "测试字符串") // Literally "Test string" in Chinese characters
    )
  }
}
