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

import scala.util.hashing.byteswap64

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{Partitioner, SparkContext}

import com.linkedin.photon.ml.Types.{FeatureShardId, REId, REType, UniqueSampleId}
import com.linkedin.photon.ml.data.scoring.CoordinateDataScores
import com.linkedin.photon.ml.spark.{BroadcastLike, RDDLike}

/**
 * dataset implementation for random effect datasets:
 *
 *   activeData + passiveData = full sharded dataset
 *
 * For the cases where a random effect ID may have a lot of data (enough that it starts causing trouble running on a
 * single node), active and passive data provide tuning options. Active data is the subset of data that is used for
 * both training and scoring (when determining residuals), while passive data is used only for scoring. In the vast
 * majority of cases, all data will be active data.
 *
 * @param activeData Grouped datasets mostly to train the sharded model and score the whole dataset.
 * @param uniqueIdToRandomEffectIds Unique id to random effect id map
 * @param passiveData Flattened datasets used to score the whole dataset
 * @param passiveDataRandomEffectIds Passive data individual IDs
 * @param randomEffectType The random effect type (e.g. "memberId")
 * @param featureShardId The feature shard ID
 */
protected[ml] class RandomEffectDataset(
    val activeData: RDD[(REId, LocalDataset)],
    protected[data] val uniqueIdToRandomEffectIds: RDD[(UniqueSampleId, REId)],
    val passiveData: RDD[(UniqueSampleId, (REId, LabeledPoint))],
    val passiveDataRandomEffectIds: Broadcast[Set[REId]],
    val randomEffectType: REType,
    val featureShardId: FeatureShardId)
  extends Dataset[RandomEffectDataset]
  with RDDLike
  with BroadcastLike {

  val randomEffectIdPartitioner: Partitioner = activeData.partitioner.get
  val uniqueIdPartitioner: Partitioner = uniqueIdToRandomEffectIds.partitioner.get

  /**
   * Add residual scores to the data offsets.
   *
   * @param scores The residual scores
   * @return The dataset with updated offsets
   */
  override def addScoresToOffsets(scores: CoordinateDataScores): RandomEffectDataset = {

    val scoresGroupedByRandomEffectId = scores
      .scores
      .join(uniqueIdToRandomEffectIds)
      .map { case (uniqueId, (score, localId)) => (localId, (uniqueId, score)) }
      .groupByKey(randomEffectIdPartitioner)
      .mapValues(_.toArray.sortBy(_._1))

    val updatedActiveData = activeData
      .join(scoresGroupedByRandomEffectId)
      .mapValues { case (localData, localScore) => localData.addScoresToOffsets(localScore) }

    val updatedPassiveData = passiveData
      .join(scores.scores)
      .mapValues { case ((randomEffectId, LabeledPoint(response, features, offset, weight)), score) =>
        (randomEffectId, LabeledPoint(response, features, offset + score, weight))
      }

    update(updatedActiveData, updatedPassiveData)
  }

  /**
   * Get the Spark context.
   *
   * @return The Spark context
   */
  override def sparkContext: SparkContext = activeData.sparkContext

  /**
   * Assign a given name to [[activeData]], [[uniqueIdToRandomEffectIds]], and [[passiveData]].
   *
   * @note Not used to reference models in the logic of photon-ml, only used for logging currently.
   * @param name The parent name for all [[RDD]]s in this class
   * @return This object with the names [[activeData]], [[uniqueIdToRandomEffectIds]], and [[passiveData]] assigned
   */
  override def setName(name: String): RandomEffectDataset = {

    activeData.setName(s"$name: Active data")
    uniqueIdToRandomEffectIds.setName(s"$name: unique id to individual Id")
    passiveData.setName(s"$name: Passive data")

    this
  }

  /**
   * Set the storage level of [[activeData]], [[uniqueIdToRandomEffectIds]], and [[passiveData]], and persist
   * their values across the cluster the first time they are computed.
   *
   * @param storageLevel The storage level
   * @return This object with the storage level of [[activeData]], [[uniqueIdToRandomEffectIds]], and [[passiveData]]
   *         set
   */
  override def persistRDD(storageLevel: StorageLevel): RandomEffectDataset = {

    if (!activeData.getStorageLevel.isValid) activeData.persist(storageLevel)
    if (!uniqueIdToRandomEffectIds.getStorageLevel.isValid) uniqueIdToRandomEffectIds.persist(storageLevel)
    if (!passiveData.getStorageLevel.isValid) passiveData.persist(storageLevel)

    this
  }

  /**
   * Mark [[activeData]], [[uniqueIdToRandomEffectIds]], and [[passiveData]] as non-persistent, and remove all
   * blocks for them from memory and disk.
   *
   * @return This object with [[activeData]], [[uniqueIdToRandomEffectIds]], and [[passiveData]] marked
   *         non-persistent
   */
  override def unpersistRDD(): RandomEffectDataset = {

    if (activeData.getStorageLevel.isValid) activeData.unpersist()
    if (uniqueIdToRandomEffectIds.getStorageLevel.isValid) uniqueIdToRandomEffectIds.unpersist()
    if (passiveData.getStorageLevel.isValid) passiveData.unpersist()

    this
  }

  /**
   * Materialize [[activeData]], [[uniqueIdToRandomEffectIds]], and [[passiveData]] (Spark [[RDD]]s are lazy
   * evaluated: this method forces them to be evaluated).
   *
   * @return This object with [[activeData]], [[uniqueIdToRandomEffectIds]], and [[passiveData]] materialized
   */
  override def materialize(): RandomEffectDataset = {

    materializeOnce(activeData, uniqueIdToRandomEffectIds, passiveData)

    this
  }

  /**
   * Asynchronously delete cached copies of [[passiveDataRandomEffectIds]] on the executors.
   *
   * @return This object with [[passiveDataRandomEffectIds]] variables unpersisted
   */
  override def unpersistBroadcast(): RandomEffectDataset = {

    passiveDataRandomEffectIds.unpersist()

    this
  }

  /**
   * Update the dataset.
   *
   * @param updatedActiveData Updated active data
   * @param updatedPassiveData Updated passive data
   * @return A new updated dataset
   */
  def update(
      updatedActiveData: RDD[(REId, LocalDataset)],
      updatedPassiveData: RDD[(UniqueSampleId, (REId, LabeledPoint))]): RandomEffectDataset =

    new RandomEffectDataset(
      updatedActiveData,
      uniqueIdToRandomEffectIds,
      updatedPassiveData,
      passiveDataRandomEffectIds,
      randomEffectType,
      featureShardId)

  /**
   * Build a human-readable summary for [[RandomEffectDataset]].
   *
   * @return A summary of the object in string representation
   */
  override def toSummaryString: String = {

    val numActiveSamples = uniqueIdToRandomEffectIds.count()
    val activeSampleWeighSum = activeData.values.map(_.getWeights.map(_._2).sum).sum()
    val activeSampleResponseSum = activeData.values.map(_.getLabels.map(_._2).sum).sum()
    val numPassiveSamples = passiveData.count()
    val passiveSampleResponsesSum = passiveData.values.map(_._2.label).sum()
    val numAllSamples = numActiveSamples + numPassiveSamples
    val numActiveSamplesStats = activeData.values.map(_.numDataPoints).stats()
    val activeSamplerResponseSumStats = activeData.values.map(_.getLabels.map(_._2).sum).stats()
    val numFeaturesStats = activeData.values.map(_.numFeatures).stats()
    val numIdsWithPassiveData = passiveDataRandomEffectIds.value.size

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

object RandomEffectDataset {

  /**
   * Build the random effect dataset with the given configuration.
   *
   * @param gameDataset The RDD of [[GameDatum]] used to generate the random effect dataset
   * @param randomEffectDataConfiguration The data configuration for the random effect dataset
   * @param randomEffectPartitioner The per random effect partitioner used to generated the grouped active data
   * @return A new random effect dataset with the given configuration
   */
  protected[ml] def apply(
      gameDataset: RDD[(UniqueSampleId, GameDatum)],
      randomEffectDataConfiguration: RandomEffectDataConfiguration,
      randomEffectPartitioner: Partitioner,
      existingModelKeysRddOpt: Option[RDD[REId]]): RandomEffectDataset = {

    val randomEffectType = randomEffectDataConfiguration.randomEffectType
    val featureShardId = randomEffectDataConfiguration.featureShardId

    val gameDataPartitioner = gameDataset.partitioner.get

    val rawActiveData = generateActiveData(
      gameDataset,
      randomEffectDataConfiguration,
      randomEffectPartitioner,
      existingModelKeysRddOpt)
    val activeData = featureSelectionOnActiveData(rawActiveData, randomEffectDataConfiguration)

    val globalIdToIndividualIds = activeData
      .flatMap { case (individualId, localDataset) =>
        localDataset.getUniqueIds.map((_, individualId))
      }
      .partitionBy(gameDataPartitioner)

    val (passiveData, passiveDataRandomEffectIds) =
      generatePassiveData(gameDataset, activeData, gameDataPartitioner, randomEffectType, featureShardId)

    new RandomEffectDataset(
      activeData,
      globalIdToIndividualIds,
      passiveData,
      passiveDataRandomEffectIds,
      randomEffectType,
      featureShardId)
  }

  /**
   * Generate active data.
   *
   * @param gameDataset The input dataset
   * @param randomEffectDataConfiguration The random effect data configuration
   * @param randomEffectPartitioner A random effect partitioner
   * @return The active dataset
   */
  private def generateActiveData(
      gameDataset: RDD[(UniqueSampleId, GameDatum)],
      randomEffectDataConfiguration: RandomEffectDataConfiguration,
      randomEffectPartitioner: Partitioner,
      existingModelKeysRddOpt: Option[RDD[REId]]): RDD[(REId, LocalDataset)] = {

    val randomEffectType = randomEffectDataConfiguration.randomEffectType
    val featureShardId = randomEffectDataConfiguration.featureShardId

    val keyedRandomEffectDataset = gameDataset.map { case (uniqueId, gameData) =>
      val randomEffectId = gameData.idTagToValueMap(randomEffectType)
      val labeledPoint = gameData.generateLabeledPointWithFeatureShardId(featureShardId)
      (randomEffectId, (uniqueId, labeledPoint))
    }

    val groupedRandomEffectDataset = randomEffectDataConfiguration
      .numActiveDataPointsUpperBound
      .map { activeDataUpperBound =>
        groupDataByKeyAndSample(
          keyedRandomEffectDataset,
          randomEffectPartitioner,
          activeDataUpperBound,
          randomEffectType)
      }
      .getOrElse(keyedRandomEffectDataset.groupByKey(randomEffectPartitioner))

    randomEffectDataConfiguration
      .numActiveDataPointsLowerBound
      .map { activeDataLowerBound =>
        existingModelKeysRddOpt match {
          case Some(existingModelKeysRdd) =>
            groupedRandomEffectDataset
              .zipPartitions(existingModelKeysRdd, preservesPartitioning = true) { (dataIt, existingKeysIt) =>

              val lookupTable = existingKeysIt.toSet

              dataIt.filter { case (key, data) =>
                (data.size >= activeDataLowerBound) || !lookupTable.contains(key)
              }
            }

          case None =>
            groupedRandomEffectDataset.filter { case (_, data) =>
              data.size >= activeDataLowerBound
            }
        }
      }
      .getOrElse(groupedRandomEffectDataset)
      .mapValues(data => LocalDataset(data.toArray, isSortedByFirstIndex = false))
  }

  /**
   * Generate a dataset, grouped by random effect ID and limited to a maximum number of samples selected via reservoir
   * sampling.
   *
   * The 'Min Heap' reservoir sampling algorithm is used for two reasons:
   * 1. The exact sampling must be reproducible so that [[RDD]] partitions can be recovered
   * 2. The linear algorithm is non-trivial to combine in a distributed manner
   *
   * @param rawKeyedDataset The raw dataset, with samples keyed by random effect ID
   * @param partitioner The partitioner
   * @param sampleCap The sample cap
   * @param randomEffectType The type of random effect
   * @return An RDD of data grouped by individual ID
   */
  private def groupDataByKeyAndSample(
      rawKeyedDataset: RDD[(REId, (UniqueSampleId, LabeledPoint))],
      partitioner: Partitioner,
      sampleCap: Int,
      randomEffectType: REType): RDD[(REId, Iterable[(UniqueSampleId, LabeledPoint)])] = {

    // Helper class for defining a constant ordering between data samples (necessary for RDD re-computation)
    case class ComparableLabeledPointWithId(comparableKey: Int, uniqueId: UniqueSampleId, labeledPoint: LabeledPoint)
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

    // The reservoir sampling algorithm is fault tolerant, assuming that the uniqueId for a sample is recovered after
    // node failure. We attempt to maximize the likelihood of successful recovery through RDD replication, however there
    // is a non-zero possibility of massive failure. If this becomes an issue, we may need to resort to check-pointing
    // the raw data RDD after uniqueId assignment.
    rawKeyedDataset
      .mapValues { case (uniqueId, labeledPoint) =>
        val comparableKey = (byteswap64(randomEffectType.hashCode) ^ byteswap64(uniqueId)).hashCode()
        ComparableLabeledPointWithId(comparableKey, uniqueId, labeledPoint)
      }
      .combineByKey[MinHeapWithFixedCapacity[ComparableLabeledPointWithId]](
        createCombiner,
        mergeValue,
        mergeCombiners,
        partitioner)
      .mapValues { minHeapWithFixedCapacity =>
        val count = minHeapWithFixedCapacity.getCount
        val data = minHeapWithFixedCapacity.getData
        val weightMultiplierOpt = if (count > sampleCap) Some(1D * count / sampleCap) else None

        data.map { case ComparableLabeledPointWithId(_, uniqueId, LabeledPoint(label, features, offset, weight)) =>
          (uniqueId, LabeledPoint(label, features, offset, weightMultiplierOpt.map(_ * weight).getOrElse(weight)))
        }
      }
  }

  /**
   * Generate passive dataset.
   *
   * @param gameDataset The raw input dataset
   * @param activeData The active dataset
   * @param gameDataPartitioner A global partitioner
   * @param randomEffectType The corresponding random effect type of the dataset
   * @param featureShardId Key of the feature shard used to generate the dataset
   * @return The passive dataset and set of random effects with passive data
   */
  private def generatePassiveData(
      gameDataset: RDD[(UniqueSampleId, GameDatum)],
      activeData: RDD[(REId, LocalDataset)],
      gameDataPartitioner: Partitioner,
      randomEffectType: REType,
      featureShardId: FeatureShardId): (RDD[(UniqueSampleId, (REId, LabeledPoint))], Broadcast[Set[REId]]) = {

    // The remaining data not included in the active data will be kept as passive data
    val activeDataUniqueIds = activeData.flatMapValues(_.dataPoints.map(_._1)).map(_.swap)
    val keyedRandomEffectDataset = gameDataset.mapValues { gameData =>
      val randomEffectId = gameData.idTagToValueMap(randomEffectType)
      val labeledPoint = gameData.generateLabeledPointWithFeatureShardId(featureShardId)
      (randomEffectId, labeledPoint)
    }

    val passiveData = keyedRandomEffectDataset.subtractByKey(activeDataUniqueIds, gameDataPartitioner)
    val passiveDataRandomEffectIds: Set[REId] = passiveData
      .map { case (_, (randomEffectId, _)) =>
        randomEffectId
      }
      .distinct()
      .collect()
      .toSet
    val passiveDataRandomEffectIdsBroadcast = gameDataset.sparkContext.broadcast(passiveDataRandomEffectIds)

    (passiveData, passiveDataRandomEffectIdsBroadcast)
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
      activeData: RDD[(REId, LocalDataset)],
      randomEffectDataConfiguration: RandomEffectDataConfiguration): RDD[(REId, LocalDataset)] = {

    randomEffectDataConfiguration
      .numFeaturesToSamplesRatioUpperBound
      .map { numFeaturesToSamplesRatioUpperBound =>
        activeData.mapValues { localDataset =>
          var numFeaturesToKeep = math.ceil(numFeaturesToSamplesRatioUpperBound * localDataset.numDataPoints).toInt

          // In case the above product overflows
          if (numFeaturesToKeep < 0) numFeaturesToKeep = Int.MaxValue
          val filteredLocalDataset = localDataset.filterFeaturesByPearsonCorrelationScore(numFeaturesToKeep)

          filteredLocalDataset
        }
      }
      .getOrElse(activeData)
  }
}
