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

import scala.util.Random

import breeze.linalg.{DenseVector, SparseVector, Vector}
import org.apache.spark.storage.StorageLevel
import org.testng.Assert._
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.Types.{FeatureShardId, REId, REType, UniqueSampleId}
import com.linkedin.photon.ml.data.scoring.CoordinateDataScores
import com.linkedin.photon.ml.projector.LinearSubspaceProjector
import com.linkedin.photon.ml.test.SparkTestUtils
import com.linkedin.photon.ml.util.LongHashPartitioner

/**
 * Integration tests for [[RandomEffectDataset]].
 */
class RandomEffectDatasetIntegTest extends SparkTestUtils {

  import RandomEffectDatasetIntegTest._

  @DataProvider
  def rawDataProvider(): Array[Array[Any]] = {

    val reId1: REId = "1"
    val reId2: REId = "2"
    val reId3: REId = "3"
    // Counts: 1 * reId1, 2 * reId2, 3 * reId3
    val dataIds: Seq[REId] = Seq(reId1, reId2, reId2, reId3, reId3, reId3)

    val data: Seq[(REId, (UniqueSampleId, LabeledPoint))] = dataIds
      .zipWithIndex
      .map { case (rEID, uID) =>
        val point = new LabeledPoint(DUMMY_RESPONSE, DUMMY_FEATURE_VECTOR)

        (rEID, (uID.toLong, point))
      }
    val partitionMap: Map[REId, Int] = Map(reId1 -> 0, reId2 -> 0, reId3 -> 0)
    val existingIds = Seq(reId1)

    val randomEffectDataConfiguration =
      RandomEffectDataConfiguration(RANDOM_EFFECT_TYPE, FEATURE_SHARD_NAME, NUM_PARTITIONS)

    Array(
      // No bounds on # samples
      Array(
        data,
        partitionMap,
        randomEffectDataConfiguration,
        None,
        3L),
      Array(
        data,
        partitionMap,
        randomEffectDataConfiguration,
        Some(existingIds),
        3L),

      // Upper bound on # samples
      Array(
        data,
        partitionMap,
        randomEffectDataConfiguration.copy(numActiveDataPointsUpperBound = Some(2)),
        None,
        3L),
      Array(
        data,
        partitionMap,
        randomEffectDataConfiguration.copy(numActiveDataPointsLowerBound = Some(1)),
        None,
        3L))
  }


  @DataProvider
  def activeDataProvider(): Array[Array[Any]] = {

    val reId1: REId = "1"
    val reId2: REId = "2"

    val dataArray1: Array[(UniqueSampleId, LabeledPoint)] = Array(
      (1, LabeledPoint(DUMMY_RESPONSE, DUMMY_FEATURE_VECTOR)))
    val dataArray2: Array[(UniqueSampleId, LabeledPoint)] = Array(
      (2, LabeledPoint(DUMMY_RESPONSE, DUMMY_FEATURE_VECTOR)),
      (3, LabeledPoint(DUMMY_RESPONSE, DUMMY_FEATURE_VECTOR)))
    val data: Seq[(REId, LocalDataset)] = Seq(
      (reId1, LocalDataset(dataArray1)),
      (reId2, LocalDataset(dataArray2)))

    val rePartitionMap: Map[REId, Int] = Map(reId1 -> 0, reId2 -> 0)

    val activeDataLowerBound = 2

    val existingIds = Seq(reId2)

    Array(
      // No existing models
      Array(
        data,
        rePartitionMap,
        activeDataLowerBound,
        None,
        Set(reId2)),

      // Some existing models
      Array(
        data,
        rePartitionMap,
        activeDataLowerBound,
        Some(existingIds),
        Set(reId1, reId2)))
  }

  @Test
  def testGenerateKeyedGameDataset(): Unit = sparkTest("testGenerateKeyedGameDataset") {

    val dummyUID = 123L
    val dummyREID = "abc"
    val dummyResponse: Double = 1.0
    val dummyOffset: Option[Double] = Some(2.0)
    val dummyWeight: Option[Double] = Some(3.0)
    val dummyFeatureVector: Vector[Double] = DenseVector(1, 2, 3)
    val dummyFeatures: Map[FeatureShardId, Vector[Double]] = Map(FEATURE_SHARD_NAME -> dummyFeatureVector)
    val dummyIDs: Map[REType, REId] = Map(RANDOM_EFFECT_TYPE -> dummyREID)

    val data: Seq[(UniqueSampleId, GameDatum)] =
      Seq((dummyUID, new GameDatum(dummyResponse, dummyOffset, dummyWeight, dummyFeatures, dummyIDs)))
    val dataRDD = sc.parallelize(data, NUM_PARTITIONS).partitionBy(new LongHashPartitioner(NUM_PARTITIONS))

    val randomEffectDataConfiguration = RandomEffectDataConfiguration(
      RANDOM_EFFECT_TYPE,
      FEATURE_SHARD_NAME,
      NUM_PARTITIONS)

    val keyedGameDataset = RandomEffectDataset.generateKeyedGameDataset(dataRDD, randomEffectDataConfiguration)
    val (actualREID, (actualUID, actualLabeledPoint)) = keyedGameDataset.take(1).head

    assertEquals(actualREID, dummyREID)
    assertEquals(actualUID, dummyUID)
    assertEquals(actualLabeledPoint.label, dummyResponse)
    assertEquals(actualLabeledPoint.offset, dummyOffset.get)
    assertEquals(actualLabeledPoint.weight, dummyWeight.get)
    assertEquals(actualLabeledPoint.features, dummyFeatureVector)
  }

  @Test
  def testGenerateLinearSubspaceProjectors(): Unit = sparkTest("testGenerateLinearSubspaceProjectors") {

    val rEId1 = "abc"
    val rEId2 = "xyz"

    val featuresLength = 8
    val featuresEvenFirstHalf = new SparseVector[Double](Array(0, 2), Array(1D, 1D), featuresLength)
    val featuresEvenSecondHalf = new SparseVector[Double](Array(4, 6), Array(1D, 1D), featuresLength)
    val featuresOddFirstHalf = new SparseVector[Double](Array(1, 3), Array(1D, 1D), featuresLength)
    val featuresOddSecondHalf = new SparseVector[Double](Array(5, 7), Array(1D, 1D), featuresLength)

    val evenFeatureIndices = featuresEvenFirstHalf.index.toSet.union(featuresEvenSecondHalf.index.toSet)
    val oddFeatureIndices = featuresOddFirstHalf.index.toSet.union(featuresOddSecondHalf.index.toSet)

    val keyedGameDataset: Seq[(REId, (UniqueSampleId, LabeledPoint))] = Seq(
      (rEId1, (1L, LabeledPoint(1D, featuresEvenFirstHalf))),
      (rEId1, (2L, LabeledPoint(1D, featuresEvenSecondHalf))),
      (rEId2, (3L, LabeledPoint(1D, featuresOddFirstHalf))),
      (rEId2, (4L, LabeledPoint(1D, featuresOddSecondHalf))))
    val keyedGameDatasetRDD = sc.parallelize(keyedGameDataset, NUM_PARTITIONS)

    val partitionMap: Map[REId, Int] = Map(rEId1 -> 0, rEId2 -> 0)
    val partitioner = new RandomEffectDatasetPartitioner(NUM_PARTITIONS, sc.broadcast(partitionMap))

    val projectorsMap = RandomEffectDataset
      .generateLinearSubspaceProjectors(keyedGameDatasetRDD, partitioner, None)
      .collect
      .toMap

    assertTrue(projectorsMap.contains(rEId1))
    assertTrue(projectorsMap.contains(rEId2))
    projectorsMap.foreach { case (_, linearSubspaceProjector) =>
      assertEquals(linearSubspaceProjector.originalSpaceDimension, featuresLength)
      assertEquals(linearSubspaceProjector.projectedSpaceDimension, featuresLength / 2)
    }
    assertEquals(projectorsMap(rEId1).projectedToOriginalSpaceMap.values.toSet, evenFeatureIndices)
    assertEquals(projectorsMap(rEId2).projectedToOriginalSpaceMap.values.toSet, oddFeatureIndices)
  }

  @Test
  def testGenerateProjectedDataset(): Unit = sparkTest("testGenerateProjectedDataset") {

    val rEId1 = "abc"
    val rEId2 = "xyz"

    val uid1 = 1L
    val uid2 = 2L

    val featuresVector1 = DenseVector[Double](1D, 2D, 3D, 4D)
    val featuresVector2 = DenseVector[Double](5D, 6D, 7D, 8D)

    val partitionMap: Map[REId, Int] = Map(rEId1 -> 0, rEId2 -> 0)
    val rEPartitioner = new RandomEffectDatasetPartitioner(NUM_PARTITIONS, sc.broadcast(partitionMap))

    val keyedGameDataset: Seq[(REId, (UniqueSampleId, LabeledPoint))] = Seq(
      (rEId1, (uid1, LabeledPoint(1D, featuresVector1))),
      (rEId2, (uid2, LabeledPoint(1D, featuresVector2))))
    val keyedGameDatasetRDD = sc.parallelize(keyedGameDataset, NUM_PARTITIONS)

    val evenIndices = Set[Int](0, 2)
    val oddIndices = Set[Int](1, 3)
    val linearSubspaceProjectorEven = new LinearSubspaceProjector(evenIndices, featuresVector1.length)
    val linearSubspaceProjectorOdd = new LinearSubspaceProjector(oddIndices, featuresVector2.length)
    val linearSubspaceProjectorsRDD = sc.parallelize(
      Seq((rEId1, linearSubspaceProjectorEven), (rEId2, linearSubspaceProjectorOdd)),
      NUM_PARTITIONS)

    val projectedDataset = RandomEffectDataset.generateProjectedDataset(
      keyedGameDatasetRDD,
      linearSubspaceProjectorsRDD,
      rEPartitioner)
    val projectedArray = projectedDataset.collect
    val (evenREID, (evenUID, evenLabeledPoint)) = projectedArray(0)
    val (oddREID, (oddUID, oddLabeledPoint)) = projectedArray(1)

    assertEquals(projectedDataset.count, 2)
    assertEquals(evenREID, rEId1)
    assertEquals(evenUID, uid1)
    assertEquals(evenLabeledPoint.features, DenseVector[Double](1D, 3D))
    assertEquals(oddREID, rEId2)
    assertEquals(oddUID, uid2)
    assertEquals(oddLabeledPoint.features, DenseVector[Double](6D, 8D))
  }

  @Test(dataProvider = "rawDataProvider")
  def testGenerateGroupedActiveData(
      data: Seq[(REId, (UniqueSampleId, LabeledPoint))],
      partitionMap: Map[REId, Int],
      config: RandomEffectDataConfiguration,
      existingIdsOpt: Option[Seq[REId]],
      expectedUniqueRandomEffects: Long): Unit = sparkTest("testGenerateGroupedActiveData") {

    val rdd = sc.parallelize(data, NUM_PARTITIONS)
    val partitioner = new RandomEffectDatasetPartitioner(NUM_PARTITIONS, sc.broadcast(partitionMap))

    val activeData = RandomEffectDataset.generateGroupedActiveData(rdd, config, partitioner)

    assertEquals(activeData.keys.count(), expectedUniqueRandomEffects)
    assertTrue(
      config
        .numActiveDataPointsUpperBound
        .forall { upperBound =>
          activeData
            .collect
            .forall { case (_, localDataset) =>
              localDataset.dataPoints.length <= upperBound
            }
        })
  }

  @Test(dataProvider = "activeDataProvider")
  def testFilterActiveData(
    data: Seq[(REId, LocalDataset)],
    partitionMap: Map[REId, Int],
    activeDataLowerBound: Int,
    existingIdsOpt: Option[Seq[REId]],
    expectedKeys: Set[REId]): Unit = sparkTest("testFilterActiveData") {

    val rePartitioner = new RandomEffectDatasetPartitioner(NUM_PARTITIONS, sc.broadcast(partitionMap))
    val dataRDD = sc.parallelize(data, NUM_PARTITIONS).partitionBy(rePartitioner)
    val existingIdsRDDOpt = existingIdsOpt.map { existingIds =>
      sc.parallelize(existingIds, NUM_PARTITIONS)
    }

    val activeDataLowerBound = 2

    val filteredActiveData = RandomEffectDataset.filterActiveData(dataRDD, activeDataLowerBound, existingIdsRDDOpt)
    val filteredKeys = filteredActiveData.keys.collect().toSet

    // Check that entities that do not meet the data threshold are filtered
    assertEquals(filteredKeys, expectedKeys)
  }

  @Test
  def testGenerateIdMap(): Unit = sparkTest("testBuild") {

    val dummyLabeledPoint = LabeledPoint(1D, DenseVector(1D, 2D, 3D))

    val rEId1 = "abc"
    val rEId2 = "xyz"
    val uIdsForREId1 = Array(1L, 3L, 5L)
    val uIdsForREId2 = Array(2L, 4L, 6L)
    val localDataset1 = LocalDataset(uIdsForREId1.map((_, dummyLabeledPoint)))
    val localDataset2 = LocalDataset(uIdsForREId2.map((_, dummyLabeledPoint)))

    val activeData = sc.parallelize(Array((rEId1, localDataset1), (rEId2, localDataset2)), NUM_PARTITIONS)
    val hashPartitioner = new LongHashPartitioner(NUM_PARTITIONS)
    val uIdsToREIds = RandomEffectDataset.generateIdMap(activeData, hashPartitioner)

    assertTrue(
      uIdsToREIds
        .collect
        .forall { case (uId, rEId) =>
          if (rEId == rEId1) {
            uIdsForREId1.contains(uId)
          } else {
            uIdsForREId2.contains(uId)
          }
        })
  }

  @Test
  def testGeneratePassiveData(): Unit = sparkTest("testBuild") {

    val numUIDs = 10
    val maxNumActiveUIDs = 5
    val dataUIDs = (0 until numUIDs).map(_.toLong).toSet
    val activeDataUIDs = (0 until numUIDs).map(_ => Random.nextInt(maxNumActiveUIDs).toLong).toSet

    val dummyREId = "abc"
    val dummyLabeledPoint = LabeledPoint(1D, DenseVector(1D, 2D, 3D))

    val projectedKeyedData: Seq[(REId, (UniqueSampleId, LabeledPoint))] = dataUIDs
      .map { uid =>
        (dummyREId, (uid, dummyLabeledPoint))
      }
      .toSeq
    val projectedKeyedDataset = sc.parallelize(projectedKeyedData, NUM_PARTITIONS)

    val activeUIDs: Seq[(UniqueSampleId, REId)] = activeDataUIDs.map((_, dummyREId)).toSeq
    val activeData = sc.parallelize(activeUIDs, NUM_PARTITIONS)

    val passiveData = RandomEffectDataset.generatePassiveData(projectedKeyedDataset, activeData)
    val passiveDataUIDs = passiveData.keys.collect.toSet

    assertEquals(passiveData.count, passiveDataUIDs.size)
    assertEquals(passiveDataUIDs, dataUIDs -- activeDataUIDs)
  }

  @Test
  def testAddScoresToOffsets(): Unit = sparkTest("testAddScoresToOffsets") {

    val reId1: REId = "1"
    val reId2: REId = "2"
    val reId3: REId = "3"
    // Counts: 1 * reId1, 2 * reId2, 3 * reId3
    val dataIds: Seq[REId] = Seq(reId1, reId2, reId2, reId3, reId3, reId3)

    val data: Seq[(UniqueSampleId, GameDatum)] = dataIds
      .zipWithIndex
      .map { case (reId, uid) =>
        val datum = new GameDatum(
          DUMMY_RESPONSE,
          DUMMY_OFFSET,
          DUMMY_WEIGHT,
          DUMMY_FEATURES,
          Map(RANDOM_EFFECT_TYPE -> reId))

        (uid.toLong, datum)
      }
    val hashPartitioner = new LongHashPartitioner(NUM_PARTITIONS)
    val dataRDD = sc.parallelize(data, NUM_PARTITIONS).partitionBy(hashPartitioner)

    val rePartitionMap: Map[REId, Int] = Map(reId1 -> 0, reId2 -> 0, reId3 -> 0)
    val rePartitioner = new RandomEffectDatasetPartitioner(NUM_PARTITIONS, sc.broadcast(rePartitionMap))

    val randomEffectDataConfig = RandomEffectDataConfiguration(
      RANDOM_EFFECT_TYPE,
      FEATURE_SHARD_NAME,
      NUM_PARTITIONS)
    val randomEffectDataset = RandomEffectDataset(
      dataRDD,
      None,
      randomEffectDataConfig,
      rePartitioner,
      None,
      StorageLevel.DISK_ONLY)

    val scores = dataIds
      .zipWithIndex
      .map { case (_, index) =>
        (index.toLong, Random.nextDouble)
      }
    val scoresRDD = sc.parallelize(scores, NUM_PARTITIONS).partitionBy(hashPartitioner)
    val coordinateDataScores = new CoordinateDataScores(scoresRDD)

    val modifiedData = randomEffectDataset
      .addScoresToOffsets(coordinateDataScores)
      .activeData
      .values
      .flatMap(_.dataPoints)
      .collect
      .sortBy(_._1)
      .map(_._2)

    assertTrue(modifiedData.zip(scores.map(_._2)).forall { case (labeledPoint, score) => labeledPoint.offset == score})
  }

  @Test
  def testLowerBoundSamplesNotInPassiveData(): Unit = sparkTest("testLowerBoundSamplesNotInPassiveData") {

    val reId1: REId = "1"
    val reId2: REId = "2"
    // Counts: 1 * reId1, 2 * reId2
    val dataIds: Seq[REId] = Seq(reId1, reId2, reId2)
    val activeDataLowerBound = 2

    val data: Seq[(UniqueSampleId, GameDatum)] = dataIds
      .zipWithIndex
      .map { case (reId, uid) =>
        val datum = new GameDatum(
          DUMMY_RESPONSE,
          DUMMY_OFFSET,
          DUMMY_WEIGHT,
          DUMMY_FEATURES,
          Map(RANDOM_EFFECT_TYPE -> reId))

        (uid.toLong, datum)
      }
    val hashPartitioner = new LongHashPartitioner(NUM_PARTITIONS)
    val dataRDD = sc.parallelize(data, NUM_PARTITIONS).partitionBy(hashPartitioner)

    val rePartitionMap: Map[REId, Int] = Map(reId1 -> 0, reId2 -> 0)
    val rePartitioner = new RandomEffectDatasetPartitioner(NUM_PARTITIONS, sc.broadcast(rePartitionMap))

    val randomEffectDataConfig = RandomEffectDataConfiguration(
      RANDOM_EFFECT_TYPE,
      FEATURE_SHARD_NAME,
      NUM_PARTITIONS,
      Some(activeDataLowerBound))
    val randomEffectDataset = RandomEffectDataset(
      dataRDD,
      None,
      randomEffectDataConfig,
      rePartitioner,
      None,
      StorageLevel.DISK_ONLY)
    val activeDataREIDs = randomEffectDataset.activeData.keys.collect().toSet

    // Check that active data lower bound filtering worked
    assertFalse(activeDataREIDs.contains(reId1))
    assertTrue(activeDataREIDs.contains(reId2))
    // Check that active data filtered due to lower bound did not end up in the passive data
    assertFalse(randomEffectDataset.passiveDataREIds.value.contains(reId1))
  }
}

object RandomEffectDatasetIntegTest {

  private val NUM_PARTITIONS = 1
  private val FEATURE_SHARD_NAME = "shard"
  private val RANDOM_EFFECT_TYPE = "reId"

  private val DUMMY_RESPONSE: Double = 1.0
  private val DUMMY_OFFSET: Option[Double] = None
  private val DUMMY_WEIGHT: Option[Double] = None
  private val DUMMY_FEATURE_VECTOR: Vector[Double] = DenseVector(1, 2, 3)
  private val DUMMY_FEATURES: Map[FeatureShardId, Vector[Double]] = Map(FEATURE_SHARD_NAME -> DUMMY_FEATURE_VECTOR)
}
