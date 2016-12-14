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

import java.io.{File, FileOutputStream}

import scala.collection.mutable.ArrayBuffer
import scala.io.Source

import org.apache.hadoop.conf.Configuration
import org.testng.Assert
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.test.TestTemplateWithTmpDir

/**
 * This class tests [[IOUtils]] can correctly read and write primitive/ASCII strings and
 * international/UTF-8 strings.
 */
class IOUtilsTest extends TestTemplateWithTmpDir {

  @DataProvider
  def dataProvider(): Array[Array[Any]] = {
    Array(
      Array("ASCII", "Test string"),
      Array("UTF-8", "テスト"), // "Test" in Japanese katakana
      Array("UTF-8", "飛行場") // "Airport" in Japanese
    )
  }

  @Test(dataProvider = "dataProvider")
  def testReadAndWrite(dir: String, testString: String): Unit = {

    val tmpDir = getTmpDir + "/" + dir
    val conf = new Configuration()

    IOUtils.writeStringsToHDFS(List(testString).iterator, tmpDir, conf, forceOverwrite = true)
    val strings = IOUtils.readStringsFromHDFS(tmpDir, conf)

    Assert.assertEquals(strings, ArrayBuffer(testString))

    new File("/tmp/test1").delete
  }

  @Test
  def testWriteToStreamSuccess(): Unit = {

    val res = IOUtils.toStream(new FileOutputStream(new File("/tmp/test1")))
      { writer => (1 to 3).foreach { i => writer.println(s"$i ") } }

    assert(res.isSuccess)
    assert(Source.fromFile("/tmp/test1").getLines.mkString == "1 2 3 ")

    new File("/tmp/test1").delete
  }

  @Test
  def testWriteToStreamFailureInOpen(): Unit = {

    val res = IOUtils.toStream(throw new RuntimeException("exception in open"))
      { writer => (1 to 3).foreach{i => writer.println(s"$i ")}}

    assert(res.isFailure)
  }

  @Test
  def testWriteToStreamFailureInWrite(): Unit = {

    val res = IOUtils.toStream(new FileOutputStream(new File("/tmp/test2")))
      { _ => throw new RuntimeException("exception in write")}

    assert(res.isFailure)

    new File("/tmp/test2").delete
  }
}
