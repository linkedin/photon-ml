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
package com.linkedin.photon.ml

import scala.collection.mutable

import com.linkedin.paldb.api.PalDB
import org.apache.commons.io.FileUtils
import org.apache.hadoop.fs.Path
import org.apache.spark.SparkException
import org.testng.Assert._
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.data.avro.ResponsePredictionFieldNames
import com.linkedin.photon.ml.test.SparkTestUtils
import com.linkedin.photon.ml.util.{IndexMap, PalDBIndexMap}

/**
 * This class tests [[com.linkedin.photon.ml.FeatureIndexingJob]].
 */
class FeatureIndexingJobTest extends SparkTestUtils {

  import FeatureIndexingJobTest._

  @DataProvider
  def partitionsAndIntercept(): Array[Array[Any]] =
    (for (partitions <- 1 to 4; intercept <- Seq(true, false)) yield {
      Array[Any](partitions, intercept)
    }).toArray

  /**
   * Test that a [[PalDBIndexMap]] can be correctly generated and registered for a data set with a single feature shard,
   * with or without intercept, using one or more partitions.
   *
   * @param partitionNum The number of partitions for the PalDB index
   * @param addIntercept Whether to add a feature for the intercept
   */
  @Test(dataProvider = "partitionsAndIntercept")
  def testIndexingJobWithPalDB(partitionNum: Int, addIntercept: Boolean): Unit = sparkTest("testIndexingJobWithPalDB") {

    val outputDir = new java.io.File(FileUtils.getTempDirectory, "test-feature-indexing-job")
    val outputDirStr = outputDir.toString

    outputDir.mkdirs()

    new FeatureIndexingJob(
      sc,
      Seq(INPUT_PATH_BASIC),
      partitionNum,
      outputDirStr,
      addIntercept,
      ResponsePredictionFieldNames)
      .run()

    // Add all partitions to cache
    for (i <- 0 until partitionNum) {
      sc.addFile(new Path(outputDirStr, PalDBIndexMap.partitionFilename(i)).toString)
    }

    val expectedFeatureDimension = if (addIntercept) 14 else 13
    val indexMap = checkPalDBReadable(
      outputDirStr,
      partitionNum,
      IndexMap.GLOBAL_NS,
      addIntercept,
      expectedFeatureDimension)

    checkHeartIndexMap(indexMap, addIntercept, expectedFeatureDimension)

    FileUtils.deleteQuietly(outputDir)
  }

  /**
   * Validate a [[PalDBIndexMap]] constructed for the heart data set.
   *
   * @param indexMap The [[PalDBIndexMap]]
   * @param addIntercept Whether a feature was included for the intercept
   * @param expectedFeatureDimension The expected feature dimension
   */
  private def checkHeartIndexMap(indexMap: PalDBIndexMap, addIntercept: Boolean, expectedFeatureDimension: Int) = {

    // Check full hits according to the ground truth of heart dataset
    val indicesSet = mutable.Set[Int]()
    val namesSet = mutable.Set[String]()
    (1 to 13).foreach{i =>
      val idx = indexMap.getIndex(i + Constants.DELIMITER)
      val name = indexMap.getFeatureName(idx).get
      assertNotEquals(idx, IndexMap.NULL_KEY)
      assertNotNull(name)
      indicesSet += idx
      namesSet += name
    }
    // Intercept
    if (addIntercept) {
      val idx = indexMap.getIndex(Constants.INTERCEPT_KEY)
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

  /**
   * Test that a [[PalDBIndexMap]] can be correctly generated and registered for a data set with a multiple feature
   * shards, with or without intercept, using one or more partitions.
   *
   * @param partitionNum The number of partitions for the PalDB index
   * @param addIntercept Whether to add a feature for the intercept
   */
  @Test(dataProvider = "partitionsAndIntercept")
  def testShardedIndexingJobWithPalDB(partitionNum: Int, addIntercept: Boolean): Unit =
    sparkTest("testShardedIndexingJobWithPalDB") {

      val outputDir = new java.io.File(FileUtils.getTempDirectory, "test-feature-indexing-job-namespaced")
      val outputDirStr = outputDir.toString

      outputDir.mkdirs()

      new FeatureIndexingJob(
        sc,
        Seq(INPUT_PATH_SHARDED),
        partitionNum,
        outputDirStr,
        addIntercept,
        ResponsePredictionFieldNames,
        Some(FEATURE_SHARD_TO_SECTION_KEYS_MAP),
        Some(FEATURE_SHARD_TO_INTERCEPT_MAP))
        .run()

      // Add all partitions to cache
      (0 until partitionNum).foreach { i =>
          FEATURE_SHARD_TO_SECTION_KEYS_MAP.foreach { case (shardId, _) =>
            sc.addFile(new Path(outputDirStr, PalDBIndexMap.partitionFilename(i, shardId)).toString)
          }
        }

      // Check each feature shard map
      FEATURE_SHARD_TO_SECTION_KEYS_MAP.foreach { case (shardId, _) =>
        val addShardIntercept = FEATURE_SHARD_TO_INTERCEPT_MAP.getOrElse(shardId, true)
        var expectedFeatureDimension = FEATURE_SHARD_TO_EXPECTED_DIMENSION(shardId)
        if (addShardIntercept) {
          expectedFeatureDimension += 1
        }

        checkPalDBReadable(outputDirStr, partitionNum, shardId, addShardIntercept, expectedFeatureDimension)
      }

      FileUtils.deleteQuietly(outputDir)
    }

  /**
   * Read and validate a [[PalDBIndexMap]].
   *
   * @param path The path to the root dir of [[PalDBIndexMap]] instances
   * @param numPartitions The number of partitions per [[PalDBIndexMap]]
   * @param namespace The name of the shard to validate
   * @param addIntercept Whether a feature was added for the intercept
   * @param expectedFeatureDimension The expected feature dimension
   * @return A [[PalDBIndexMap]]
   */
  private def checkPalDBReadable(
    path: String,
    numPartitions: Int,
    namespace: String,
    addIntercept: Boolean,
    expectedFeatureDimension: Int): PalDBIndexMap = {

    val indexMap = new PalDBIndexMap().load(path, numPartitions, namespace)

    var offset = 0
    for (i <- 0 until numPartitions) {
      val reader = PalDB.createReader(new java.io.File(path, PalDBIndexMap.partitionFilename(i, namespace)))
      val iter = reader.iterable().iterator()

      while (iter.hasNext) {
        val entry = iter.next()

        val key = entry.getKey.asInstanceOf[Object]
        val value = entry.getValue.asInstanceOf[Object]

        key match {
          case s: String if value.isInstanceOf[Int] =>
            val result = reader.get(s).asInstanceOf[Int]
            val idx = indexMap.getIndex(s)

            // What IndexMap read and what the original data file reads are consistent
            assertEquals(value, result)
            assertEquals(idx, result + offset)
          case _ =>
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

    indexMap
  }

  /**
   * Test that attempts to generate a [[PalDBIndexMap]] for a non-existant or poorly defined feature shard will fail.
   *
   * @param partitionNum The number of partitions for the PalDB index
   * @param addIntercept Whether to add a feature for the intercept
   */
  @Test(dataProvider = "partitionsAndIntercept", expectedExceptions = Array(classOf[SparkException]))
  def testBadShardedIndexingJobWithPalDB(partitionNum: Int, addIntercept: Boolean): Unit =
    sparkTest("testShardedIndexingJobWithPalDB") {

      val outputDir = new java.io.File(FileUtils.getTempDirectory, "test-feature-indexing-job-namespaced")
      val outputDirStr = outputDir.toString

      outputDir.mkdirs()

      new FeatureIndexingJob(
        sc,
        Seq(INPUT_PATH_SHARDED),
        partitionNum,
        outputDirStr,
        addIntercept,
        ResponsePredictionFieldNames,
        Some(FEATURE_SHARD_TO_SECTION_KEYS_MAP ++ Map("shard3" -> Set("badsection"))),
        Some(FEATURE_SHARD_TO_INTERCEPT_MAP))
        .run()
    }
}

object FeatureIndexingJobTest {

  val INPUT_PATH_BASIC = new Path("src/integTest/resources/DriverIntegTest/input/heart.avro")
  val INPUT_PATH_SHARDED = new Path("src/integTest/resources/GameIntegTest/input/train/yahoo-music-train.avro")

  val FEATURE_SHARD_TO_SECTION_KEYS_MAP = Map(
    "shard1" -> Set("features", "songFeatures"),
    "shard2" -> Set("songFeatures")
  )
  val FEATURE_SHARD_TO_INTERCEPT_MAP = Map(
    "shard1" -> true,
    "shard2" -> false
  )
  val FEATURE_SHARD_TO_EXPECTED_DIMENSION = Map(
    "shard1" -> 15014,
    "shard2" -> 30
  )
}
