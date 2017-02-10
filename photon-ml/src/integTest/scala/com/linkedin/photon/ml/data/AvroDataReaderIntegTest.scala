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
package com.linkedin.photon.ml.data

import breeze.stats._
import org.apache.spark.SparkException
import org.apache.spark.mllib.linalg.{SparseVector, Vectors}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.testng.Assert._
import org.testng.annotations.Test

import com.linkedin.photon.ml.constants.MathConst
import com.linkedin.photon.ml.test.SparkTestUtils
import com.linkedin.photon.ml.util._

class AvroDataReaderIntegTest extends SparkTestUtils {
  val tol = MathConst.HIGH_PRECISION_TOLERANCE_THRESHOLD
  val inputPath = getClass.getClassLoader.getResource("GameIntegTest/input/train").getPath
  val inputPath2 = getClass.getClassLoader.getResource("GameIntegTest/input/test").getPath
  val duplicateFeaturesPath = getClass.getClassLoader.getResource("GameIntegTest/input/duplicateFeatures").getPath
  val indexMapPath = getClass.getClassLoader.getResource("GameIntegTest/input/feature-indexes").getPath
  val featureSectionMap = Map(
    "shard1" -> Set("userFeatures", "songFeatures"),
    "shard2" -> Set("userFeatures"),
    "shard3" -> Set("songFeatures")
  )
  val numPartitions = 4

  @Test
  def testRead(): Unit = sparkTest("testRead") {
    val dr = new AvroDataReader(sc)
    val (df, _) = dr.readMerged(inputPath, featureSectionMap, numPartitions)

    verifyDataFrame(df, expectedRows = 34810)
  }

  @Test
  def testReadWithFeatureIndex(): Unit = sparkTest("testReadWithIndex") {
    val indexMapParams = new PalDBIndexMapParams {
      offHeapIndexMapDir = Some(indexMapPath)
    }

    val indexMapLoaders = featureSectionMap.map { case (shardId, _) => {
      val indexMapLoader = new PalDBIndexMapLoader
      indexMapLoader.prepare(sc, indexMapParams, shardId)
      (shardId, indexMapLoader)
    }}

    val dr = new AvroDataReader(sc)
    val df = dr.readMerged(inputPath, indexMapLoaders, featureSectionMap, numPartitions)

    verifyDataFrame(df, expectedRows = 34810)
  }

  @Test
  def testReadMultipleFiles(): Unit = sparkTest("testReadMultipleFiles") {
    val dr = new AvroDataReader(sc)
    val (df, _) = dr.readMerged(Seq(inputPath, inputPath2), featureSectionMap, numPartitions)

    verifyDataFrame(df, expectedRows = 44005)
  }

  @Test
  def testReadDuplicateFeatures(): Unit = sparkTest("testReadDuplicateFeatures") {
    val dr = new AvroDataReader(sc)

    try {
      val (df, _) = dr.read(duplicateFeaturesPath, numPartitions)

      // Force evaluation
      df.head

      fail("Expected failure didn't happen.")

    } catch {
      case se: SparkException => assertTrue(se.getMessage.contains("Duplicate features found"))
      case e: Exception => throw(e)
    }
  }

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testReadInvalidPaths(): Unit = sparkTest("testReadInvalidPaths") {
    val dr = new AvroDataReader(sc)
    dr.read(Seq.empty[String], numPartitions)
  }

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testReadInvalidPartitions(): Unit = sparkTest("testReadInvalidPartitions") {
    val dr = new AvroDataReader(sc)
    dr.read(inputPath, -1)
  }

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testInvalidInputPath(): Unit = sparkTest("testInvalidInputPath") {
    val emptyInputPath = getClass.getClassLoader.getResource("GameIntegTest/empty-input").getPath
    val dr = new AvroDataReader(sc)
    dr.read(emptyInputPath, numPartitions)
  }

  /**
   * Verifies that the DataFrame has expected shape and statistics.
   *
   * Note: the baseline statistics here were computed manually from the raw data.
   *
   * @param df the DataFrame to test
   * @param expectedRows the number of rows the DataFrame should have
   */
  def verifyDataFrame(df: DataFrame, expectedRows: Int): Unit = {
    // Columns have the expected number of features and summary stats look right
    assertTrue(df.columns.contains("shard1"))
    val vector1 = df.select(col("shard1")).take(1)(0).getAs[SparseVector](0)
    assertEquals(vector1.numActives, 61)
    assertEquals(Vectors.norm(vector1, 2), 3.2298996752519407, tol)
    val (mu1: Double, _, var1: Double) = DescriptiveStats.meanAndCov(vector1.values, vector1.values)
    assertEquals(mu1, 0.044020727910406766, tol)
    assertEquals(var1, 0.17190074364268512, tol)

    assertTrue(df.columns.contains("shard2"))
    val vector2 = df.select(col("shard2")).take(1)(0).getAs[SparseVector](0)
    assertEquals(vector2.numActives, 31)
    assertEquals(Vectors.norm(vector2, 2), 2.509607963949448, tol)
    val (mu2: Double, _, var2: Double) = DescriptiveStats.meanAndCov(vector2.values, vector2.values)
    assertEquals(mu2, 0.05196838235602745, tol)
    assertEquals(var2, 0.20714700123375754, tol)

    assertTrue(df.columns.contains("shard3"))
    val vector3 = df.select(col("shard3")).take(1)(0).getAs[SparseVector](0)
    assertEquals(vector3.numActives, 31)
    assertEquals(Vectors.norm(vector3, 2), 2.265859611598675, tol)
    val (mu3: Double, _, var3: Double) = DescriptiveStats.meanAndCov(vector3.values, vector3.values)
    assertEquals(mu3, 0.06691111449993427, tol)
    assertEquals(var3, 0.16651099216405915, tol)

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
