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

import scala.collection.Set
import scala.util.hashing.byteswap64

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.{StorageLevel => SparkStorageLevel}
import org.apache.spark.{Partitioner, SparkContext}

import com.linkedin.photon.ml.constants.StorageLevel
import com.linkedin.photon.ml.{BroadcastLike, RDDLike}

/**
 * Data set implementation for random effect datasets:
 *
 *   passiveData + activeData = full sharded data set
 *
 * @param activeData Grouped data sets mostly to train the sharded model and score the whole data set.
 * @param uniqueIdToRandomEffectIds Unique id to random effect id map
 * @param passiveDataOption Flattened data sets used to score the whole data set
 * @param passiveDataRandomEffectIdsOption passive data individual IDs
 * @param randomEffectType The random effect type (e.g. "memberId")
 * @param featureShardId The feature shard ID
 */
protected[ml] class RandomEffectDataSet(
    val activeData: RDD[(String, LocalDataSet)],
    protected[data] val uniqueIdToRandomEffectIds: RDD[(Long, String)],
    val passiveDataOption: Option[RDD[(Long, (String, LabeledPoint))]],
    val passiveDataRandomEffectIdsOption: Option[Broadcast[Set[String]]],
    val randomEffectType: String,
    val featureShardId: String)
  extends DataSet[RandomEffectDataSet]
  with RDDLike
  with BroadcastLike {

  val randomEffectIdPartitioner = activeData.partitioner.get
  val uniqueIdPartitioner = uniqueIdToRandomEffectIds.partitioner.get
  val hasPassiveData = passiveDataOption.isDefined

  override def addScoresToOffsets(scores: KeyValueScore): RandomEffectDataSet = {
    val scoresGroupedByRandomEffectId = scores
      .scores
      .join(uniqueIdToRandomEffectIds)
      .map { case (uniqueId, (score, localId)) => (localId, (uniqueId, score)) }
      .groupByKey(randomEffectIdPartitioner)
      .mapValues(_.toArray.sortBy(_._1))

    val updatedActiveData = activeData
      .join(scoresGroupedByRandomEffectId)
      .mapValues { case (localData, localScore) => localData.addScoresToOffsets(localScore) }

    val updatedPassiveDataOption = passiveDataOption.map(
      _.join(scores.scores)
        .mapValues { case ((randomEffectId, LabeledPoint(response, features, offset, weight)), score) =>
          (randomEffectId, LabeledPoint(response, features, offset + score, weight))
        })

    update(updatedActiveData, updatedPassiveDataOption)
  }

  override def sparkContext: SparkContext = activeData.sparkContext

  override def setName(name: String): this.type = {
    activeData.setName(s"$name: Active data")
    uniqueIdToRandomEffectIds.setName(s"$name: unique id to individual Id")
    passiveDataOption.foreach(_.setName(s"$name: Passive data"))
    this
  }

  override def persistRDD(storageLevel: SparkStorageLevel): this.type = {
    if (!activeData.getStorageLevel.isValid) activeData.persist(storageLevel)
    if (!uniqueIdToRandomEffectIds.getStorageLevel.isValid) uniqueIdToRandomEffectIds.persist(storageLevel)
    passiveDataOption.foreach { passiveData =>
      if (!passiveData.getStorageLevel.isValid) passiveData.persist(storageLevel)
    }
    this
  }

  override def unpersistRDD(): this.type = {
    if (activeData.getStorageLevel.isValid) activeData.unpersist()
    if (uniqueIdToRandomEffectIds.getStorageLevel.isValid) uniqueIdToRandomEffectIds.unpersist()
    passiveDataOption.foreach { passiveData =>
      if (passiveData.getStorageLevel.isValid) passiveData.unpersist()
    }
    this
  }

  override def unpersistBroadcast(): this.type = {
    passiveDataRandomEffectIdsOption.foreach(_.unpersist())
    this
  }

  override def materialize(): this.type = {
    activeData.count()
    uniqueIdToRandomEffectIds.count()
    passiveDataOption.foreach(_.count())
    this
  }

  /**
   * Update the dataset
   *
   * @param updatedActiveData Updated active data
   * @param updatedPassiveDataOption (Optional) Updated passive data
   * @return A new updated dataset
   */
  def update(
      updatedActiveData: RDD[(String, LocalDataSet)],
      updatedPassiveDataOption: Option[RDD[(Long, (String, LabeledPoint))]]): RandomEffectDataSet = {
    new RandomEffectDataSet(
      updatedActiveData,
      uniqueIdToRandomEffectIds,
      updatedPassiveDataOption,
      passiveDataRandomEffectIdsOption,
      randomEffectType,
      featureShardId)
  }

  override def toSummaryString: String = {
    val numActiveSamples = uniqueIdToRandomEffectIds.count()
    val activeSampleWeighSum = activeData.values.map(_.getWeights.map(_._2).sum).sum()
    val activeSampleResponseSum = activeData.values.map(_.getLabels.map(_._2).sum).sum()
    val numPassiveSamples = if (hasPassiveData) passiveDataOption.get.count() else 0
    val passiveSampleResponsesSum = if (hasPassiveData) passiveDataOption.get.values.map(_._2.label).sum() else 0
    val numAllSamples = numActiveSamples + numPassiveSamples
    val numActiveSamplesStats = activeData.values.map(_.numDataPoints).stats()
    val activeSamplerResponseSumStats = activeData.values.map(_.getLabels.map(_._2).sum).stats()
    val numFeaturesStats = activeData.values.map(_.numActiveFeatures).stats()
    val numIdsWithPassiveData =
      if (passiveDataRandomEffectIdsOption.isDefined) passiveDataRandomEffectIdsOption.get.value.size else 0

    s"numActiveSamples: $numActiveSamples\n" +
        s"activeSampleWeighSum: $activeSampleWeighSum\n" +
        s"activeSampleResponseSum: $activeSampleResponseSum\n" +
        s"numPassiveSamples: $numPassiveSamples\n" +
        s"passiveSampleResponsesSum: $passiveSampleResponsesSum\n" +
        s"numAllSamples: $numAllSamples\n" +
        s"numActiveSamplesStats: $numActiveSamplesStats\n" +
        s"activeSamplerResponseSumStats: $activeSamplerResponseSumStats\n" +
        s"numFeaturesStats: $numFeaturesStats\n" +
        s"numIdsWithPassiveData: $numIdsWithPassiveData"
  }
}

object RandomEffectDataSet {
  /**
   * Build the random effect data set with the given configuration
   *
   * @param gameDataSet The RDD of [[GameDatum]] used to generate the random effect data set
   * @param randomEffectDataConfiguration The data configuration for the random effect data set
   * @param randomEffectPartitioner The per random effect partitioner used to generated the grouped active data
   * @return A new random effect dataset with the given configuration
   */
  protected[ml] def buildWithConfiguration(
      gameDataSet: RDD[(Long, GameDatum)],
      randomEffectDataConfiguration: RandomEffectDataConfiguration,
      randomEffectPartitioner: Partitioner): RandomEffectDataSet = {

    val randomEffectType = randomEffectDataConfiguration.randomEffectType
    val featureShardId = randomEffectDataConfiguration.featureShardId

    val gameDataPartitioner = gameDataSet.partitioner.get

    val rawActiveData = generateActiveData(gameDataSet, randomEffectDataConfiguration, randomEffectPartitioner)
    val activeData = featureSelectionOnActiveData(rawActiveData, randomEffectDataConfiguration)
      .setName("Active data")
      .persist(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)

    val globalIdToIndividualIds = activeData
      .flatMap { case (individualId, localDataSet) =>
        localDataSet.getUniqueIds.map((_, individualId))
      }
      .partitionBy(gameDataPartitioner)

    val (passiveDataOption, passiveDataRandomEffectIdsOption) =
      if (randomEffectDataConfiguration.isDownSamplingNeeded) {
        val (passiveData, passiveDataRandomEffectIds) =
          generatePassiveData(gameDataSet, activeData, gameDataPartitioner, randomEffectDataConfiguration)
        (Option(passiveData), Option(passiveDataRandomEffectIds))
      } else {
        (None, None)
      }

    new RandomEffectDataSet(
      activeData,
      globalIdToIndividualIds,
      passiveDataOption,
      passiveDataRandomEffectIdsOption,
      randomEffectType,
      featureShardId)
  }

  /**
   * Generate active data
   *
   * @param gameDataSet The input dataset
   * @param randomEffectDataConfiguration The random effect data configuration
   * @param randomEffectPartitioner A random effect partitioner
   * @return The active dataset
   */
  private def generateActiveData(
      gameDataSet: RDD[(Long, GameDatum)],
      randomEffectDataConfiguration: RandomEffectDataConfiguration,
      randomEffectPartitioner: Partitioner): RDD[(String, LocalDataSet)] = {

    val randomEffectType = randomEffectDataConfiguration.randomEffectType
    val featureShardId = randomEffectDataConfiguration.featureShardId
    val numActiveDataPointsToKeepUpperBound = randomEffectDataConfiguration.numActiveDataPointsToKeepUpperBound

    val keyedRandomEffectDataSet = gameDataSet.map { case (uniqueId, gameData) =>
      val randomEffectId = gameData.idTypeToValueMap(randomEffectType)
      val labeledPoint = gameData.generateLabeledPointWithFeatureShardId(featureShardId)
      (randomEffectId, (uniqueId, labeledPoint))
    }

    val groupedRandomEffectDataSet =
      if (randomEffectDataConfiguration.isDownSamplingNeeded) {
        groupKeyedDataSetViaReservoirSampling(
          keyedRandomEffectDataSet,
          randomEffectPartitioner,
          numActiveDataPointsToKeepUpperBound,
          randomEffectType)
      } else {
        keyedRandomEffectDataSet.groupByKey(randomEffectPartitioner)
      }

    groupedRandomEffectDataSet.mapValues(iterable => LocalDataSet(iterable.toArray, isSortedByFirstIndex = false))
  }

  /**
   * Generate a group keyed dataset via reservoir sampling
   *
   * @param rawKeyedDataSet The raw keyed dataset
   * @param partitioner The partitioner
   * @param sampleCap The sample cap
   * @param randomEffectType The type of random effect
   * @return An RDD of data grouped by individual ID
   */
  private def groupKeyedDataSetViaReservoirSampling(
      rawKeyedDataSet: RDD[(String, (Long, LabeledPoint))],
      partitioner: Partitioner,
      sampleCap: Int,
      randomEffectType: String): RDD[(String, Iterable[(Long, LabeledPoint)])] = {

    case class ComparableLabeledPointWithId(comparableKey: Int, uniqueId: Long, labeledPoint: LabeledPoint)
      extends Comparable[ComparableLabeledPointWithId] {

      override def compareTo(comparableLabeledPointWithId: ComparableLabeledPointWithId): Int = {
        if (comparableKey - comparableLabeledPointWithId.comparableKey > 0) {
          1
        } else {
          -1
        }
      }
    }

    val createCombiner =
      (comparableLabeledPointWithId: ComparableLabeledPointWithId) => {
        new MinHeapWithFixedCapacity[ComparableLabeledPointWithId](sampleCap) += comparableLabeledPointWithId
      }

    val mergeValue = (
        minHeapWithFixedCapacity: MinHeapWithFixedCapacity[ComparableLabeledPointWithId],
        comparableLabeledPointWithId: ComparableLabeledPointWithId) => {
      minHeapWithFixedCapacity += comparableLabeledPointWithId
    }

    val mergeCombiners = (
        minHeapWithFixedCapacity1: MinHeapWithFixedCapacity[ComparableLabeledPointWithId],
        minHeapWithFixedCapacity2: MinHeapWithFixedCapacity[ComparableLabeledPointWithId]) => {
      minHeapWithFixedCapacity1 ++= minHeapWithFixedCapacity2
    }

    /*
     * Currently the reservoir sampling algorithm is not fault tolerant, as the comparable key depends on uniqueId,
     * which is computed based on the RDD partition id. The uniqueId will change after recomputing the RDD while
     * recovering from node failure.
     *
     * TODO: Need to make sure that the comparableKey is robust to RDD recompute and node failure
     */
    val localDataSets =
      rawKeyedDataSet
        .mapValues { case (uniqueId, labeledPoint) =>
          val comparableKey = (byteswap64(randomEffectType.hashCode) ^ byteswap64(uniqueId)).hashCode()
          ComparableLabeledPointWithId(comparableKey, uniqueId, labeledPoint)
        }
        .combineByKey[MinHeapWithFixedCapacity[ComparableLabeledPointWithId]](createCombiner, mergeValue,
           mergeCombiners, partitioner)
        .mapValues { minHeapWithFixedCapacity =>
          val cumCount = minHeapWithFixedCapacity.cumCount
          val data = minHeapWithFixedCapacity.getData
          val size = data.size
          val weightMultiplierFactor = 1.0 * cumCount / size
          val dataPoints =
            data.map { case ComparableLabeledPointWithId(_, uniqueId, LabeledPoint(label, features, offset, weight)) =>
              (uniqueId, LabeledPoint(label, features, offset, weight * weightMultiplierFactor))
            }
          dataPoints
        }

    localDataSets
  }

  /**
   * Generate passive dataset
   *
   * @param gameDataSet The input dataset
   * @param activeData The active dataset
   * @param gameDataPartitioner A global paritioner
   * @param randomEffectDataConfiguration The random effect data configuration
   * @return The passive dataset
   */
  private def generatePassiveData(
      gameDataSet: RDD[(Long, GameDatum)],
      activeData: RDD[(String, LocalDataSet)],
      gameDataPartitioner: Partitioner,
      randomEffectDataConfiguration: RandomEffectDataConfiguration):
    (RDD[(Long, (String, LabeledPoint))], Broadcast[Set[String]]) = {

    val randomEffectType = randomEffectDataConfiguration.randomEffectType
    val featureShardId = randomEffectDataConfiguration.featureShardId
    val numPassiveDataPointsToKeepLowerBound = randomEffectDataConfiguration.numPassiveDataPointsToKeepLowerBound

    // The remaining data not included in the active data will be kept as passive data
    val activeDataUniqueIds = activeData.flatMapValues(_.dataPoints.map(_._1)).map(_.swap)
    val keyedRandomEffectDataSet = gameDataSet.mapValues { gameData =>
      val randomEffectId = gameData.idTypeToValueMap(randomEffectType)
      val labeledPoint = gameData.generateLabeledPointWithFeatureShardId(featureShardId)
      (randomEffectId, labeledPoint)
    }

    val passiveData = keyedRandomEffectDataSet.subtractByKey(activeDataUniqueIds, gameDataPartitioner)
        .setName("tmp passive data")
        .persist(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)

    val passiveDataRandomEffectIdCountsMap = passiveData.map { case (_, (randomEffectId, _)) => (randomEffectId, 1) }
        .reduceByKey(_ + _).collectAsMap()

    // Only keep the passive data whose total number of data points is larger than the given lower bound
    val passiveDataRandomEffectIds = passiveDataRandomEffectIdCountsMap
        .filter(_._2 > numPassiveDataPointsToKeepLowerBound).keySet
    val sparkContext = gameDataSet.sparkContext
    val passiveDataRandomEffectIdsBroadcast = sparkContext.broadcast(passiveDataRandomEffectIds)
    val filteredPassiveData = passiveData.filter { case ((_, (id, _))) =>
      passiveDataRandomEffectIdsBroadcast.value.contains(id)
      }
      .setName("passive data")
      .persist(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)

    filteredPassiveData.count()
    passiveData.unpersist()

    (filteredPassiveData, passiveDataRandomEffectIdsBroadcast)
  }

  /**
   * Reduce active data feature dimension for individuals with few samples. The maximum feature dimension is limited to
   * the number of samples multiplied by the feature dimension ratio. Features are chosen by greatest Pearson
   * correlation score.
   *
   * @param activeData The active dataset
   * @param randomEffectDataConfiguration The random effect data configuration
   * @return The active data with the feature dimension reduced to the maximum
   */
  private def featureSelectionOnActiveData(
      activeData: RDD[(String, LocalDataSet)],
      randomEffectDataConfiguration: RandomEffectDataConfiguration): RDD[(String, LocalDataSet)] = {

    val numFeaturesToSamplesRatioUpperBound = randomEffectDataConfiguration.numFeaturesToSamplesRatioUpperBound
    activeData.mapValues { localDataSet =>
      var numFeaturesToKeep = math.ceil(numFeaturesToSamplesRatioUpperBound * localDataSet.numDataPoints).toInt

      // In case the above product overflows
      if (numFeaturesToKeep < 0) numFeaturesToKeep = Int.MaxValue
      val filteredLocalDataSet = localDataSet.filterFeaturesByPearsonCorrelationScore(numFeaturesToKeep)

      filteredLocalDataSet
    }
  }
}
