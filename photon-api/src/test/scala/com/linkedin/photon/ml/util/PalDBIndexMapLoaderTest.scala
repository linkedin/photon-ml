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

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.mockito.Mockito._
import org.testng.Assert._
import org.testng.annotations.{DataProvider, Test}

/**
 *
 */
class PalDBIndexMapLoaderTest {

  @DataProvider
  def pathDataProvider(): Array[Array[String]] = {
    Array(
      Array("hdfs:/", "/path/to/resource", "hdfs:/path/to/resource"),
      Array("hdfs:/", "s3a://path/to/resource", "s3a://path/to/resource"),
      Array("hdfs://myhost:9199", "/path/to/resource", "hdfs://myhost:9199/path/to/resource"),
      Array("s3a://bucket", "/path/to/resource", "s3a://bucket/path/to/resource"))
  }

  @Test(dataProvider = "pathDataProvider")
  def testGetPath(defaultName: String, inputPath: String, expectedResult: String): Unit = {
    val sc = mock(classOf[SparkContext])
    val conf = mock(classOf[Configuration])
    val path = new Path(inputPath)

    doReturn(conf).when(sc).hadoopConfiguration
    doReturn(defaultName).when(conf).get("fs.default.name")

    assertEquals(PalDBIndexMapLoader.getPath(sc, path), expectedResult)
  }
}
