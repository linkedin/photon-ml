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
package com.linkedin.photon.ml.data

import breeze.linalg.{DenseVector, Vector}
import org.testng.Assert._
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.Types.{FeatureShardId, REId, UniqueSampleId}
import com.linkedin.photon.ml.test.SparkTestUtils
import com.linkedin.photon.ml.util.LongHashPartitioner

/**
 * Integration tests for [[RandomEffectDataSet]].
 */
class RandomEffectDataSetIntegTest extends SparkTestUtils {

  import RandomEffectDataSetIntegTest._

  @DataProvider
  def rawDataProvider(): Array[Array[Any]] = {

    val dummyResponse: Double = 1.0
    val dummyOffset: Option[Double] = None
    val dummyWeight: Option[Double] = None
    val dummyFeatureVector: Vector[Double] = DenseVector(1, 2, 3)
    val dummyFeatures: Map[FeatureShardId, Vector[Double]] = Map(FEATURE_SHARD_NAME -> dummyFeatureVector)

    val reId1: REId = "1"
    val reId2: REId = "2"
    val reId3: REId = "3"
    // Counts: 1 * reId1, 2 * reId2, 3 * reId3
    val dataIds: Seq[REId] = Seq(reId1, reId2, reId2, reId3, reId3, reId3)

    val data: Seq[(UniqueSampleId, GameDatum)] = dataIds
      .zipWithIndex
      .map { case (reId, uid) =>
        val datum = new GameDatum(
          dummyResponse,
          dummyOffset,
          dummyWeight,
          dummyFeatures,
          Map(RANDOM_EFFECT_TYPE -> reId))

        (uid.toLong, datum)
      }
    val partitionMap: Map[REId, Int] = Map(reId1 -> 0, reId2 -> 0, reId3 -> 0)

    Array(
      Array(data, partitionMap, 1, 3L),
      Array(data, partitionMap, 2, 2L),
      Array(data, partitionMap, 3, 1L))
  }

  @DataProvider
  def rawDataProvider2(): Array[Array[Any]] = {

    val dummyResponse: Double = 1.0
    val dummyOffset: Option[Double] = None
    val dummyWeight: Option[Double] = None
    val dummyFeatureVector: Vector[Double] = DenseVector(1, 2, 3)
    val dummyFeatures: Map[FeatureShardId, Vector[Double]] = Map(FEATURE_SHARD_NAME -> dummyFeatureVector)

    val reId1: REId = "1"
    val reId2: REId = "2"
    val reId3: REId = "3"
    // Counts: 1 * reId1, 2 * reId2, 3 * reId3
    val dataIds: Seq[REId] = Seq(reId1, reId2, reId2, reId3, reId3, reId3)

    val data: Seq[(UniqueSampleId, GameDatum)] = dataIds
      .zipWithIndex
      .map { case (reId, uid) =>
        val datum = new GameDatum(
          dummyResponse,
          dummyOffset,
          dummyWeight,
          dummyFeatures,
          Map(RANDOM_EFFECT_TYPE -> reId))

        (uid.toLong, datum)
      }
    val existingIds = Seq(reId1)
    val partitionMap: Map[REId, Int] = Map(reId1 -> 0, reId2 -> 0, reId3 -> 0)

    Array(
      Array(data, existingIds, partitionMap, 1, 3L),
      Array(data, existingIds, partitionMap, 2, 2L),
      Array(data, existingIds, partitionMap, 3, 2L))
  }

  /**
   * Test that the random effects with less data than the active data lower bound will be dropped.
   *
   * @param data Raw training data
   * @param partitionMap Raw map to build [[RandomEffectDataSetPartitioner]]
   * @param activeDataLowerBound Threshold for active data
   * @param expectedUniqueRandomEffects Expected number of random effects which have data exceeding the threshold
   */
  @Test(dataProvider = "rawDataProvider")
  def testActiveDataLowerBound(
      data: Seq[(UniqueSampleId, GameDatum)],
      partitionMap: Map[REId, Int],
      activeDataLowerBound: Int,
      expectedUniqueRandomEffects: Long): Unit = sparkTest("testActiveDataLowerBound") {

    val rdd = sc.parallelize(data, NUM_PARTITIONS).partitionBy(new LongHashPartitioner(NUM_PARTITIONS))
    val randomEffectDataConfig = RandomEffectDataConfiguration(
      RANDOM_EFFECT_TYPE,
      FEATURE_SHARD_NAME,
      NUM_PARTITIONS,
      Some(activeDataLowerBound))
    val partitioner = new RandomEffectDataSetPartitioner(NUM_PARTITIONS, sc.broadcast(partitionMap))

    val randomEffectDataSet = RandomEffectDataSet(rdd, randomEffectDataConfig, partitioner, None, None)
    val numUniqueRandomEffects = randomEffectDataSet.activeData.keys.count()

    assertEquals(numUniqueRandomEffects, expectedUniqueRandomEffects)
  }

  /**
   * Test that the random effects with less data than the active data lower bound will be dropped.
   *
   * @param data Raw training data
   * @param partitionMap Raw map to build [[RandomEffectDataSetPartitioner]]
   * @param activeDataLowerBound Threshold for active data
   * @param expectedUniqueRandomEffects Expected number of random effects which have data exceeding the threshold
   */
  @Test(dataProvider = "rawDataProvider2")
  def testIgnoreActiveDataLowerBound(
      data: Seq[(UniqueSampleId, GameDatum)],
      existingIds: Seq[REId],
      partitionMap: Map[REId, Int],
      activeDataLowerBound: Int,
      expectedUniqueRandomEffects: Long): Unit = sparkTest("testActiveDataLowerBound") {

    val rdd = sc.parallelize(data, NUM_PARTITIONS).partitionBy(new LongHashPartitioner(NUM_PARTITIONS))
    val existingIdsRDD = sc.parallelize(existingIds, NUM_PARTITIONS)

    val randomEffectDataConfig = RandomEffectDataConfiguration(
      RANDOM_EFFECT_TYPE,
      FEATURE_SHARD_NAME,
      NUM_PARTITIONS,
      Some(activeDataLowerBound))
    val partitioner = new RandomEffectDataSetPartitioner(NUM_PARTITIONS, sc.broadcast(partitionMap))

    val randomEffectDataSet = RandomEffectDataSet(rdd, randomEffectDataConfig, partitioner, Some(existingIdsRDD), None)
    val numUniqueRandomEffects = randomEffectDataSet.activeData.keys.count()

    assertEquals(numUniqueRandomEffects, expectedUniqueRandomEffects)
  }
}

object RandomEffectDataSetIntegTest {

  private val NUM_PARTITIONS = 1
  private val FEATURE_SHARD_NAME = "shard"
  private val RANDOM_EFFECT_TYPE = "reId"
}
