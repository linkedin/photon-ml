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
package com.linkedin.photon.ml.data.avro

import breeze.stats._
import org.apache.hadoop.fs.Path
import org.apache.spark.SparkException
import org.apache.spark.ml.linalg.{SparseVector, Vectors}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.testng.Assert._
import org.testng.annotations.Test

import com.linkedin.photon.ml.Constants
import com.linkedin.photon.ml.index. PalDBIndexMapLoader
import com.linkedin.photon.ml.io.FeatureShardConfiguration
import com.linkedin.photon.ml.test.{CommonTestUtils, SparkTestUtils}

/**
 * Unit tests for AvroDataReader
 */
class AvroDataReaderIntegTest extends SparkTestUtils {

  import AvroDataReaderIntegTest._

  /**
   * Test reading avro data with mixed value types in map
   */
  @Test
  def testAvroWithVariousTypeMap(): Unit = sparkTest("avroMapTest") {
    val dr = new AvroDataReader()
    val dataPath = new Path(INPUT_DIR, "avroMap")
    val featureConfigMap = Map("shard1" -> FeatureShardConfiguration(Set("xgboost_click"), hasIntercept = true))
    val (df, _) = dr.readMerged(dataPath.toString, featureConfigMap, None)

    assertEquals(df.count(), 2)
  }

  /**
   * Test reading a [[DataFrame]].
   */
  @Test
  def testRead(): Unit = sparkTest("testRead") {
    val dr = new AvroDataReader()
    val (df, _) = dr.readMerged(TRAIN_INPUT_PATH.toString, FEATURE_SHARD_CONFIGS_MAP, NUM_PARTITIONS)

    verifyDataFrame(df, expectedRows = 34810)
  }

  /**
   * Test reading a [[DataFrame]], using an existing [[com.linkedin.photon.ml.index.IndexMap]].
   */
  @Test
  def testReadWithFeatureIndex(): Unit = sparkTest("testReadWithIndex") {
    val indexMapLoaders = FEATURE_SHARD_CONFIGS_MAP.map { case (shardId, _) =>
      val indexMapLoader = PalDBIndexMapLoader(
        sc,
        INDEX_MAP_PATH,
        numPartitions = 1,
        shardId)

      (shardId, indexMapLoader)
    }

    val dr = new AvroDataReader()
    val df = dr.readMerged(
      TRAIN_INPUT_PATH.toString,
      indexMapLoaders,
      FEATURE_SHARD_CONFIGS_MAP,
      NUM_PARTITIONS)

    verifyDataFrame(df, expectedRows = 34810)
  }

  /**
   * Test reading a [[DataFrame]] from multiple paths.
   */
  @Test
  def testReadMultipleFiles(): Unit = sparkTest("testReadMultipleFiles") {
    val dr = new AvroDataReader()
    val (df, _) = dr.readMerged(
      Seq(TRAIN_INPUT_PATH, TEST_INPUT_PATH).map(_.toString),
      FEATURE_SHARD_CONFIGS_MAP,
      NUM_PARTITIONS)

    verifyDataFrame(df, expectedRows = 44005)
  }

  /**
   * Test reading a [[DataFrame]] without intercepts.
   */
  @Test
  def testNoIntercept(): Unit = sparkTest("testNoIntercept") {

    val shardId = "shard2"
    val dr = new AvroDataReader()
    val modifiedFeatureShardConfigsMap = Map(
      shardId -> FEATURE_SHARD_CONFIGS_MAP(shardId).copy(hasIntercept = false))
    val (df, indexMapLoaders) = dr.readMerged(
      TRAIN_INPUT_PATH.toString,
      modifiedFeatureShardConfigsMap,
      NUM_PARTITIONS)

    // Assert that shard2 exists and has the correct # features
    assertTrue(df.columns.contains(shardId))
    assertEquals(df.select(col(shardId)).take(1)(0).getAs[SparseVector](0).numActives, 30)

    // Assert that the intercept is not in the IndexMap
    assertFalse(indexMapLoaders(shardId).indexMapForDriver().contains(Constants.INTERCEPT_KEY))

    // Assert that all rows have been read
    assertEquals(df.count, 34810)
  }

  /**
   * Test that reading a [[DataFrame]] with duplicate features will throw an error.
   */
  @Test(expectedExceptions = Array(classOf[SparkException]))
  def testReadDuplicateFeatures(): Unit = sparkTest("testReadDuplicateFeatures") {
    val dr = new AvroDataReader()
    val (df, _) = dr.read(DUPLICATE_FEATURES_PATH.toString, NUM_PARTITIONS)

    // Force evaluation
    df.head
  }

  /**
   * Test that reading a [[DataFrame]] from an invalid path will throw an error.
   */
  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testReadInvalidPaths(): Unit = sparkTest("testReadInvalidPaths") {
    val dr = new AvroDataReader()
    dr.read(Seq.empty[String], NUM_PARTITIONS)
  }

  /**
   * Test that attempting to create a [[DataFrame]] with an invalid number of partitions will throw an error.
   */
  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testReadInvalidPartitions(): Unit = sparkTest("testReadInvalidPartitions") {
    val dr = new AvroDataReader()
    dr.read(TRAIN_INPUT_PATH.toString, Some(-1))
  }

  /**
   * Test that reading a [[DataFrame]] from files without data will throw an error.
   */
  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testInvalidInputPath(): Unit = sparkTest("testInvalidInputPath") {
    val emptyInputPath = getClass.getClassLoader.getResource("GameIntegTest/empty-input").getPath
    val dr = new AvroDataReader()
    dr.read(emptyInputPath, NUM_PARTITIONS)
  }
}

object AvroDataReaderIntegTest {

  private val INPUT_DIR = getClass.getClassLoader.getResource("GameIntegTest/input").getPath
  private val TRAIN_INPUT_PATH = new Path(INPUT_DIR, "train")
  private val TEST_INPUT_PATH = new Path(INPUT_DIR, "test")
  private val DUPLICATE_FEATURES_PATH = new Path(INPUT_DIR, "duplicateFeatures")
  private val INDEX_MAP_PATH = new Path(INPUT_DIR, "feature-indexes")
  private val NUM_PARTITIONS = Some(4)
  private val FEATURE_SHARD_CONFIGS_MAP = Map(
    "shard1" -> FeatureShardConfiguration(Set("userFeatures", "songFeatures"), hasIntercept = true),
    "shard2" -> FeatureShardConfiguration(Set("userFeatures"), hasIntercept = true),
    "shard3" -> FeatureShardConfiguration(Set("songFeatures"), hasIntercept = true))

  /**
   * Verifies that the DataFrame has expected shape and statistics.
   *
   * @note The baseline statistics here were computed manually from the raw data.
   *
   * @param df The DataFrame to test
   * @param expectedRows The number of rows the DataFrame should have
   */
  def verifyDataFrame(df: DataFrame, expectedRows: Int): Unit = {

    // Columns have the expected number of features and summary stats look right
    assertTrue(df.columns.contains("shard1"))
    val vector1 = df.select(col("shard1")).take(1)(0).getAs[SparseVector](0)
    assertEquals(vector1.numActives, 61)
    assertEquals(Vectors.norm(vector1, 2), 3.2298996752519407, CommonTestUtils.HIGH_PRECISION_TOLERANCE)
    val (mu1: Double, _, var1: Double) = DescriptiveStats.meanAndCov(vector1.values, vector1.values)
    assertEquals(mu1, 0.044020727910406766, CommonTestUtils.HIGH_PRECISION_TOLERANCE)
    assertEquals(var1, 0.17190074364268512, CommonTestUtils.HIGH_PRECISION_TOLERANCE)

    assertTrue(df.columns.contains("shard2"))
    val vector2 = df.select(col("shard2")).take(1)(0).getAs[SparseVector](0)
    assertEquals(vector2.numActives, 31)
    assertEquals(Vectors.norm(vector2, 2), 2.509607963949448, CommonTestUtils.HIGH_PRECISION_TOLERANCE)
    val (mu2: Double, _, var2: Double) = DescriptiveStats.meanAndCov(vector2.values, vector2.values)
    assertEquals(mu2, 0.05196838235602745, CommonTestUtils.HIGH_PRECISION_TOLERANCE)
    assertEquals(var2, 0.20714700123375754, CommonTestUtils.HIGH_PRECISION_TOLERANCE)

    assertTrue(df.columns.contains("shard3"))
    val vector3 = df.select(col("shard3")).take(1)(0).getAs[SparseVector](0)
    assertEquals(vector3.numActives, 31)
    assertEquals(Vectors.norm(vector3, 2), 2.265859611598675, CommonTestUtils.HIGH_PRECISION_TOLERANCE)
    val (mu3: Double, _, var3: Double) = DescriptiveStats.meanAndCov(vector3.values, vector3.values)
    assertEquals(mu3, 0.06691111449993427, CommonTestUtils.HIGH_PRECISION_TOLERANCE)
    assertEquals(var3, 0.16651099216405915, CommonTestUtils.HIGH_PRECISION_TOLERANCE)

    // Relationship between columns is the same across the entire dataframe
    df.foreach { row =>
      val shard1 = row.getAs[SparseVector]("shard1")
      val shard2 = row.getAs[SparseVector]("shard2")
      val shard3 = row.getAs[SparseVector]("shard3")

      // It's (n - 1) because each column has an intercept
      assertEquals(shard1.numActives, shard2.numActives + shard3.numActives - 1)
    }

    // Source columns have been removed
    assertTrue(!df.columns.contains("userFeatures"))
    assertTrue(!df.columns.contains("songFeatures"))

    assertEquals(df.count, expectedRows)
  }
}
