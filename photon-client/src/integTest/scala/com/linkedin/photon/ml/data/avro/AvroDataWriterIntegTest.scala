/*
 * Copyright 2018 LinkedIn Corp. All rights reserved.
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
package com.linkedin.photon.ml.data.avro

import org.apache.hadoop.fs.{FileSystem, Path}
import org.testng.Assert._
import org.testng.annotations.Test

import com.linkedin.photon.ml.io.FeatureShardConfiguration
import com.linkedin.photon.ml.test.{SparkTestUtils, TestTemplateWithTmpDir}

/**
 * Integration tests for [[AvroDataWriter]].
 */
class AvroDataWriterIntegTest extends SparkTestUtils with TestTemplateWithTmpDir {

  import AvroDataWriterIntegTest._

  @Test
  def testWrite(): Unit = sparkTest("testRead") {
    val dr = new AvroDataReader()
    val (df, indexMapLoadersMap) = dr.readMerged(INPUT_PATH.toString, FEATURE_SHARD_CONFIGURATIONS_MAP, NUM_PARTITIONS)
    val outputDir = new Path(getTmpDir)

    assertTrue(df.columns.contains(FEATURE_COLUMN))
    assertTrue(df.columns.contains(RESPONSE_COLUMN))
    assertEquals(df.count, 34810)
    assertTrue(indexMapLoadersMap.contains(FEATURE_COLUMN))

    val indexMapLoader = indexMapLoadersMap(FEATURE_COLUMN)
    val writer = new AvroDataWriter
    writer.write(df, outputDir.toString, indexMapLoader, RESPONSE_COLUMN, FEATURE_COLUMN, overwrite = true)

    val fs = FileSystem.get(sc.hadoopConfiguration)
    val files = fs.listStatus(outputDir).filter(_.getPath.getName.startsWith("part"))
    assertEquals(files.length, NUM_PARTITIONS.get)

    val (writeData, _) = dr.read(outputDir.toString, NUM_PARTITIONS)
    assertTrue(writeData.columns.contains(RESPONSE_COLUMN))
    assertTrue(writeData.columns.contains(FEATURE_COLUMN))
    assertEquals(writeData.count(), 34810)
  }
}

object AvroDataWriterIntegTest {
  private val INPUT_DIR = getClass.getClassLoader.getResource("GameIntegTest/input").getPath
  private val INPUT_PATH = new Path(INPUT_DIR, "train")
  private val NUM_PARTITIONS = Some(4)
  private val FEATURE_COLUMN = "features"
  private val RESPONSE_COLUMN = "response"
  private val FEATURE_SHARD_CONFIGURATIONS_MAP = Map(
    FEATURE_COLUMN -> FeatureShardConfiguration(Set("userFeatures", "songFeatures"), hasIntercept = false))
}
