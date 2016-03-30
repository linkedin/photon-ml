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
package com.linkedin.photon.ml

import com.linkedin.paldb.api.PalDB
import com.linkedin.photon.ml.avro.{ResponsePredictionFieldNames, FieldNames}
import com.linkedin.photon.ml.avro.model.TrainingExampleFieldNames
import com.linkedin.photon.ml.io.GLMSuite
import com.linkedin.photon.ml.test.SparkTestUtils
import com.linkedin.photon.ml.util.{IndexMap, PalDBIndexMap}
import org.apache.commons.io.FileUtils
import org.apache.hadoop.fs.Path
import org.apache.spark.{SparkConf, SparkContext}
import org.testng.Assert._
import org.testng.annotations.Test

import scala.collection.mutable


/**
  * This class tests [[com.linkedin.photon.ml.FeatureIndexingJob]]
  *
  */
class FeatureIndexingJobTest {

  @Test
  def testIndexingJobWithPalDB(): Unit = {
    var partitionNum: Int = 1

    val tempFile = new java.io.File(FileUtils.getTempDirectory, "test-feature-indexing-job")
    tempFile.mkdirs()

    for (partitionNum <- 1 to 4) {
      testOneJob(tempFile.toString(), partitionNum, true, TrainingExampleFieldNames)
      testOneJob(tempFile.toString(), partitionNum, false, TrainingExampleFieldNames)
      testOneJob(tempFile.toString(), partitionNum, true, ResponsePredictionFieldNames)
      testOneJob(tempFile.toString(), partitionNum, false, ResponsePredictionFieldNames)
    }

    FileUtils.deleteQuietly(tempFile)
  }

  private def testOneJob(outputDir: String = "/tmp/index-output",
                         numPartitions: Int,
                         addIntercept: Boolean,
                         fieldNames: FieldNames): Unit = {
    SparkTestUtils.SPARK_LOCAL_CONFIG.synchronized {
      FileUtils.deleteDirectory(new java.io.File(outputDir))

      val conf: SparkConf = new SparkConf()
      conf.setAppName("test-index-job").setMaster("local[4]")
      val sc = new SparkContext(conf)

      try {
        new FeatureIndexingJob(sc,
          "src/integTest/resources/DriverIntegTest/input/heart.avro",
          numPartitions,
          outputDir,
          addIntercept,
          fieldNames
        ).run()

        // Add all partitions to cache
        (0 until numPartitions).foreach(i =>
            sc.addFile(new Path(outputDir, PalDBIndexMap.partitionFilename(i)).toString()))

        checkPalDBReadable(outputDir, numPartitions, addIntercept)
      } finally {
        sc.stop()
        System.clearProperty("spark.driver.port")
        System.clearProperty("spark.hostPort")
      }
    }
  }

  private def checkPalDBReadable(path: String, numPartitions: Int, addIntercept: Boolean): Unit = {
    val indexMap = new PalDBIndexMap().load(path, numPartitions)

    val expectedFeatureDimension = if (addIntercept) 14 else 13

    var i = 0
    var offset = 0
    for (i <- 0 until numPartitions) {
      val reader = PalDB.createReader(new java.io.File(path, PalDBIndexMap.partitionFilename(i)))
      val iter = reader.iterable().iterator()

      while (iter.hasNext) {
        val entry = iter.next()

        val key = entry.getKey().asInstanceOf[Object];
        val value = entry.getValue().asInstanceOf[Object];

        if (key.isInstanceOf[String] && value.isInstanceOf[Int]) {
          val result = reader.get(key.asInstanceOf[String]).asInstanceOf[Int]
          val idx = indexMap.getIndex(key.asInstanceOf[String])

          // What IndexMap read and what the original data file reads are consistent
          assertEquals(value, result)
          assertEquals(idx, result + offset)
        } else {
          val featureName = indexMap.getFeatureName(key.asInstanceOf[Int] + offset).get
          // The index map itself should be consistent
          assertEquals(indexMap.getIndex(featureName), key.asInstanceOf[Int] + offset)
        }
      }

      offset += reader.size().asInstanceOf[Number].intValue()/2
      reader.close()
    }

    // Check IndexMap itself
    println("Total record number: " + indexMap.size())
    assertEquals(indexMap.size(), expectedFeatureDimension)

    // Check full hits according to the ground truth of heart dataset
    val indicesSet = mutable.Set[Int]()
    val namesSet = mutable.Set[String]()
    (1 to 13).foreach{i =>
      val idx = indexMap.getIndex(i + GLMSuite.DELIMITER)
      val name = indexMap.getFeatureName(idx).get
      assertNotEquals(idx, IndexMap.NULL_KEY)
      assertNotNull(name)
      indicesSet += idx
      namesSet += name
    }
    // Intercept
    if (addIntercept) {
      val idx = indexMap.getIndex(GLMSuite.INTERCEPT_NAME_TERM)
      val name = indexMap.getFeatureName(idx).get
      assertNotEquals(idx, IndexMap.NULL_KEY)
      assertNotNull(name)
      indicesSet += idx
      namesSet += name
    }
    assertEquals(indicesSet.size, expectedFeatureDimension)
    (0 to 12).foreach(i => assertTrue(indicesSet.contains(i)))
    assertEquals(namesSet.size, expectedFeatureDimension)

    // Check other indices or string aren't there
    assertEquals(indexMap.getIndex(""), IndexMap.NULL_KEY)
    assertEquals(indexMap.getIndex("sdfds"), IndexMap.NULL_KEY)
    assertTrue(indexMap.getFeatureName(expectedFeatureDimension).isEmpty)
    assertTrue(indexMap.getFeatureName(IndexMap.NULL_KEY).isEmpty)

    // Check iterator
    indicesSet.clear()
    namesSet.clear()
    val iter = indexMap.iterator
    var count = 0
    while (iter.hasNext) {
      val entry: (String, Int) = iter.next()

      val idx = indexMap.getIndex(entry._1)
      assertEquals(idx, indexMap.get(entry._1).get)
      assertEquals(idx, entry._2)
      assertEquals(entry._1, indexMap.getFeatureName(idx).get)
      indicesSet += idx
      namesSet += entry._1
    }
    assertEquals(indicesSet.size, expectedFeatureDimension)
    assertEquals(namesSet.size, expectedFeatureDimension)
  }
}
