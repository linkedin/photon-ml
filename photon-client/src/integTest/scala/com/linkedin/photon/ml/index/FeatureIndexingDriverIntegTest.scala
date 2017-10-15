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
package com.linkedin.photon.ml.index

import scala.collection.mutable

import com.linkedin.paldb.api.PalDB
import org.apache.hadoop.fs.Path
import org.apache.spark.SparkException
import org.apache.spark.ml.param.{Param, ParamMap}
import org.testng.Assert._
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.Constants
import com.linkedin.photon.ml.io.FeatureShardConfiguration
import com.linkedin.photon.ml.test.{SparkTestUtils, TestTemplateWithTmpDir}

/**
 * Integration tests for [[FeatureIndexingDriver]].
 */
class FeatureIndexingDriverIntegTest extends SparkTestUtils with TestTemplateWithTmpDir {

  import FeatureIndexingDriverIntegTest._

  @DataProvider
  def partitionsAndIntercept(): Array[Array[Any]] =
    (for (partitions <- 1 to 4) yield {
      Array[Any](partitions)
    })
    .toArray

  /**
   * Test that a [[PalDBIndexMap]] can be correctly generated and registered for a data set with a single feature shard,
   * with intercept, using one or more partitions.
   *
   * @param partitionNum The number of partitions for the PalDB index
   */
  @Test(dataProvider = "partitionsAndIntercept")
  def testIndexingJobWithPalDB(partitionNum: Int): Unit = sparkTest("testIndexingJobWithPalDB") {

    val outputPath = new Path(getTmpDir)
    val params = heartArgs
      .put(FeatureIndexingDriver.rootOutputDirectory, outputPath)
      .put(FeatureIndexingDriver.numPartitions, partitionNum)

    runDriver(params)

    // Add all partitions to cache
    for (i <- 0 until partitionNum) {
      sc.addFile(new Path(outputPath, PalDBIndexMap.partitionFilename(i)).toString)
    }

    val indexMap = checkPalDBReadable(
      outputPath.toString,
      partitionNum,
      heartGlobalFeatureShardId,
      HEART_EXPECTED_FEATURE_DIM)

    checkHeartIndexMap(indexMap, HEART_EXPECTED_FEATURE_DIM)
  }

  /**
   * Validate a [[PalDBIndexMap]] constructed for the heart data set.
   *
   * @param indexMap The [[PalDBIndexMap]]
   * @param expectedFeatureDimension The expected feature dimension
   */
  private def checkHeartIndexMap(indexMap: PalDBIndexMap, expectedFeatureDimension: Int) = {

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
    val idx = indexMap.getIndex(Constants.INTERCEPT_KEY)
    val name = indexMap.getFeatureName(idx).get
    assertNotEquals(idx, IndexMap.NULL_KEY)
    assertNotNull(name)
    indicesSet += idx
    namesSet += name
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
   */
  @Test(dataProvider = "partitionsAndIntercept")
  def testShardedIndexingJobWithPalDB(partitionNum: Int): Unit =
    sparkTest("testShardedIndexingJobWithPalDB") {

      val outputPath = new Path(getTmpDir)
      val params = yahooMusicArgs
        .put(FeatureIndexingDriver.rootOutputDirectory, outputPath)
        .put(FeatureIndexingDriver.numPartitions, partitionNum)
      val featureShardConfigs = params.get(FeatureIndexingDriver.featureShardConfigurations).get

      runDriver(params)

      // Add all partitions to cache
      (0 until partitionNum).foreach { i =>
        featureShardConfigs.foreach { case (featureShardId, _) =>
          sc.addFile(new Path(outputPath, PalDBIndexMap.partitionFilename(i, featureShardId)).toString)
        }
      }

      // Check each feature shard map
      featureShardConfigs.foreach { case (featureShardId, _) =>
        val expectedFeatureDimension = YAHOO_MUSIC_EXPECTED_FEATURE_DIM(featureShardId)

        checkPalDBReadable(outputPath.toString, partitionNum, featureShardId, expectedFeatureDimension)
      }
    }

  /**
   * Read and validate a [[PalDBIndexMap]].
   *
   * @param path The path to the root dir of [[PalDBIndexMap]] instances
   * @param numPartitions The number of partitions per [[PalDBIndexMap]]
   * @param namespace The name of the shard to validate
   * @param expectedFeatureDimension The expected feature dimension
   * @return A [[PalDBIndexMap]]
   */
  private def checkPalDBReadable(
    path: String,
    numPartitions: Int,
    namespace: String,
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
   */
  @Test(expectedExceptions = Array(classOf[SparkException]))
  def testBadShardedIndexingJobWithPalDB(): Unit = sparkTest("testShardedIndexingJobWithPalDB") {

    val badFeatureShardConfigs = Map(
      "badShard" -> FeatureShardConfiguration(Set("fakeFeatureBag"), hasIntercept = false))
    val outputPath = new Path(getTmpDir)
    val params = yahooMusicArgs
      .put(FeatureIndexingDriver.rootOutputDirectory, outputPath)
      .put(FeatureIndexingDriver.numPartitions, 1)
      .put(FeatureIndexingDriver.featureShardConfigurations, badFeatureShardConfigs)

    runDriver(params)
  }

  /**
   * Run the feature indexing driver with the specified arguments.
   *
   * @param params Arguments for feature indexing
   */
  def runDriver(params: ParamMap): Unit = {

    // Reset Driver parameters
    FeatureIndexingDriver.clear()

    params
      .toSeq
      .foreach(paramPair => FeatureIndexingDriver.set(paramPair.param.asInstanceOf[Param[Any]], paramPair.value))

    FeatureIndexingDriver.sc = sc
    FeatureIndexingDriver.run()
  }
}

object FeatureIndexingDriverIntegTest {

  val heartInputPaths = Set(new Path("src/integTest/resources/DriverIntegTest/input/heart.avro"))
  val yahooMusicInputPaths = Set(new Path("src/integTest/resources/GameIntegTest/input/train/yahoo-music-train.avro"))

  val heartGlobalFeatureShardId = "global"
  val heartFeatureBags = Set("features")
  val heartGlobalFeatureShardConfig = FeatureShardConfiguration(heartFeatureBags, hasIntercept = true)
  val heartFeatureShardConfigs = Map(heartGlobalFeatureShardId -> heartGlobalFeatureShardConfig)

  val yahooMusicGlobalFeatureShardId = "global"
  val yahooMusicGlobalFeatureBags = Set("features", "songFeatures")
  val yahooMusicGlobalIntercept = true
  val yahooMusicGlobalFeatureShardConfig = FeatureShardConfiguration(yahooMusicGlobalFeatureBags, yahooMusicGlobalIntercept)
  val yahooMusicSongFeatureShardId = "song"
  val yahooMusicSongFeatureBags = Set("songFeatures")
  val yahooMusicSongIntercept = false
  val yahooMusicSongFeatureShardConfig = FeatureShardConfiguration(yahooMusicSongFeatureBags, yahooMusicSongIntercept)
  val yahooMusicFeatureShardConfigs = Map(
    yahooMusicGlobalFeatureShardId -> yahooMusicGlobalFeatureShardConfig,
    yahooMusicSongFeatureShardId -> yahooMusicSongFeatureShardConfig)

  val HEART_EXPECTED_FEATURE_DIM = 14
  val YAHOO_MUSIC_EXPECTED_FEATURE_DIM = Map(
    yahooMusicGlobalFeatureShardId -> 15015,
    yahooMusicSongFeatureShardId -> 30)

  /**
   * Default arguments to the feature indexing driver.
   *
   * @return Arguments to prepare a feature index map
   */
  private def defaultArgs: ParamMap = ParamMap
    .empty
    .put(FeatureIndexingDriver.overrideOutputDirectory, true)

  /**
   * Heart data set arguments to the feature indexing driver.
   *
   * @return Arguments to prepare a feature index map
   */
  private def heartArgs: ParamMap = defaultArgs
    .put(FeatureIndexingDriver.inputDataDirectories, heartInputPaths)
    .put(FeatureIndexingDriver.featureShardConfigurations, heartFeatureShardConfigs)

  /**
   * Yahoo music data set arguments to the feature indexing driver.
   *
   * @return Arguments to prepare a feature index map
   */
  private def yahooMusicArgs: ParamMap = defaultArgs
    .put(FeatureIndexingDriver.inputDataDirectories, yahooMusicInputPaths)
    .put(FeatureIndexingDriver.featureShardConfigurations, yahooMusicFeatureShardConfigs)
}
