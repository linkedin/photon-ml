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
import org.apache.spark.sql.SparkSession
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{Partitioner, SparkContext}

import com.linkedin.photon.ml.Types.{FeatureShardId, REId, REType, UniqueSampleId}
import com.linkedin.photon.ml.data.RandomEffectDataset.{ActiveData, PassiveData}
import com.linkedin.photon.ml.data.scoring.CoordinateDataScores
import com.linkedin.photon.ml.spark.{BroadcastLike, RDDLike}

/**
 * Dataset implementation for random effect data.
 *
 * All of the training data for a single random effect must fit into on Spark partition. The size limit of a single
 * Spark partition is 2 GB. If the size of (samples * features) exceeds the maximum size of a single Spark partition,
 * the data is split into two sections: active and passive data.
 *
 *   activeData + passiveData = full data set
 *
 * Active data is used for both training and scoring (to determine residuals for partial score). Passive data is used
 * only for scoring. In the vast majority of cases, all data is active data.
 *
 * @param activeData Per-entity datasets used to train per-entity models and to compute residuals
 * @param passiveData Per-entity datasets used only to compute residuals
 * @param activeUniqueIdToRandomEffectIds Map of unique sample id to random effect id for active data samples
 * @param randomEffectType The random effect type (e.g. "memberId")
 * @param featureShardId The ID of the data feature shard used by this dataset
 */
protected[ml] class RandomEffectDataset(
    val activeData: ActiveData,
    val passiveData: PassiveData,
    val activeUniqueIdToRandomEffectIds: RDD[(UniqueSampleId, REId)],
    val randomEffectType: REType,
    val featureShardId: FeatureShardId)
  extends Dataset[RandomEffectDataset]
    with BroadcastLike
    with RDDLike {

  lazy val passiveDataREIds: Broadcast[Set[REId]] = SparkSession
    .builder()
    .getOrCreate()
    .sparkContext
    .broadcast(passiveData.map(_._2._1).distinct().collect().toSet)
  val randomEffectIdPartitioner: Partitioner = activeData.partitioner.get
  val uniqueIdPartitioner: Partitioner = passiveData.partitioner.get

  //
  // RandomEffectDataset functions
  //

  /**
   * Create a new [[RandomEffectDataset]] with new active and passive data.
   *
   * @param newActiveData New active data [[RDD]]
   * @param newPassiveData New passive data [[RDD]]
   * @return A new [[RandomEffectDataset]]
   */
  def update(newActiveData: ActiveData, newPassiveData: PassiveData): RandomEffectDataset =
    new RandomEffectDataset(
      newActiveData,
      newPassiveData,
      // Need to clone RDD so that old dataset can be unpersisted
      activeUniqueIdToRandomEffectIds.mapPartitions(identity, preservesPartitioning = true),
      randomEffectType,
      featureShardId)

  //
  // Dataset functions
  //

  /**
   * Add residual scores to the data offsets.
   *
   * @param scores The residual scores
   * @return The dataset with updated offsets
   */
  override def addScoresToOffsets(scores: CoordinateDataScores): RandomEffectDataset = {

    // Add scores to active data offsets
    val scoresGroupedByRandomEffectId = scores
      .scoresRdd
      .join(activeUniqueIdToRandomEffectIds, uniqueIdPartitioner)
      .map { case (uniqueId, (score, reId)) => (reId, (uniqueId, score)) }
      .groupByKey(randomEffectIdPartitioner)
      .mapValues(_.toArray.sortBy(_._1))

    // Both RDDs use the same partitioner
    val updatedActiveData = activeData
      .join(scoresGroupedByRandomEffectId, randomEffectIdPartitioner)
      .mapValues { case (localData, localScore) => localData.addScoresToOffsets(localScore) }

    // The resultant dataset is only used for training a new model, thus only the active data needs to have scores added
    new RandomEffectDataset(
      updatedActiveData,
      passiveData,
      activeUniqueIdToRandomEffectIds,
      randomEffectType,
      featureShardId)
  }

  //
  // BroadcastLike Functions
  //

  /**
   * Asynchronously delete cached copies of [[passiveDataREIds]] on all executors.
   *
   * @return This [[RandomEffectDataset]] with [[passiveDataREIds]] unpersisted
   */
  override protected[ml] def unpersistBroadcast(): RandomEffectDataset = {

    passiveDataREIds.unpersist()

    this
  }

  //
  // RDDLike Functions
  //

  /**
   * Get the Spark context.
   *
   * @return The Spark context
   */
  override def sparkContext: SparkContext = activeData.sparkContext

  /**
   * Assign a given name to [[activeData]], [[activeUniqueIdToRandomEffectIds]], and [[passiveData]].
   *
   * @note Not used to reference models in the logic of photon-ml, only used for logging currently.
   * @param name The parent name for all [[RDD]]s in this class
   * @return This object with the names [[activeData]], [[activeUniqueIdToRandomEffectIds]], and [[passiveData]]
   *         assigned
   */
  override def setName(name: String): RandomEffectDataset = {

    activeData.setName(s"$name - Active Data")
    passiveData.setName(s"$name - Passive Data")
    activeUniqueIdToRandomEffectIds.setName(s"$name - UID to REID")

    this
  }

  /**
   * Set the storage level of [[activeData]], [[activeUniqueIdToRandomEffectIds]], and [[passiveData]], and persist
   * their values across the cluster the first time they are computed.
   *
   * @param storageLevel The storage level
   * @return This object with the storage level of [[activeData]], [[activeUniqueIdToRandomEffectIds]], and
   *         [[passiveData]] set
   */
  override def persistRDD(storageLevel: StorageLevel): RandomEffectDataset = {

    if (!activeData.getStorageLevel.isValid) activeData.persist(storageLevel)
    if (!passiveData.getStorageLevel.isValid) passiveData.persist(storageLevel)
    if (!activeUniqueIdToRandomEffectIds.getStorageLevel.isValid) activeUniqueIdToRandomEffectIds.persist(storageLevel)

    this
  }

  /**
   * Mark [[activeData]], [[activeUniqueIdToRandomEffectIds]], and [[passiveData]] as non-persistent, and remove all
   * blocks for them from memory and disk.
   *
   * @return This object with [[activeData]], [[activeUniqueIdToRandomEffectIds]], and [[passiveData]] marked
   *         non-persistent
   */
  override def unpersistRDD(): RandomEffectDataset = {

    if (activeData.getStorageLevel.isValid) activeData.unpersist()
    if (passiveData.getStorageLevel.isValid) passiveData.unpersist()
    if (activeUniqueIdToRandomEffectIds.getStorageLevel.isValid) activeUniqueIdToRandomEffectIds.unpersist()

    this
  }

  /**
   * Materialize [[activeData]], [[activeUniqueIdToRandomEffectIds]], and [[passiveData]] (Spark [[RDD]]s are lazy
   * evaluated: this method forces them to be evaluated).
   *
   * @return This object with [[activeData]], [[activeUniqueIdToRandomEffectIds]], and [[passiveData]] materialized
   */
  override def materialize(): RandomEffectDataset = {

    activeData.count()
    passiveData.count()
    activeUniqueIdToRandomEffectIds.count()

    this
  }

  //
  // Summarizable Functions
  //

  /**
   * Build a human-readable summary for [[RandomEffectDataset]].
   *
   * @return A summary of the object in string representation
   */
  override def toSummaryString: String = {

    val stringBuilder = new StringBuilder("Random Effect Data Set:")

    val activeDataValues = activeData.values.persist(StorageLevel.MEMORY_ONLY_SER)

    val numActiveSamples = activeUniqueIdToRandomEffectIds.count()
    val activeSampleWeightSum = activeDataValues.map(_.getWeights.map(_._2).sum).sum()
    val activeSampleResponseSum = activeDataValues.map(_.getLabels.map(_._2).sum).sum()
    val numPassiveSamples = passiveData.count()
    val passiveSampleResponsesSum = passiveData.values.map(_._2.label).sum()
    val numAllSamples = numActiveSamples + numPassiveSamples
    val numActiveSamplesStats = activeDataValues.map(_.numDataPoints).stats()
    val activeSamplerResponseSumStats = activeDataValues.map(_.getLabels.map(_._2).sum).stats()
    val numFeaturesStats = activeDataValues.map(_.numFeatures).stats()

    activeDataValues.unpersist()

    // TODO: Need more descriptive text than just the variable name
    stringBuilder.append(s"\nnumActiveSamples: $numActiveSamples")
    stringBuilder.append(s"\nactiveSampleWeightSum: $activeSampleWeightSum")
    stringBuilder.append(s"\nactiveSampleResponseSum: $activeSampleResponseSum")
    stringBuilder.append(s"\nnumPassiveSamples: $numPassiveSamples")
    stringBuilder.append(s"\npassiveSampleResponsesSum: $passiveSampleResponsesSum")
    stringBuilder.append(s"\nnumAllSamples: $numAllSamples")
    stringBuilder.append(s"\nnumActiveSamplesStats: $numActiveSamplesStats")
    stringBuilder.append(s"\nactiveSamplerResponseSumStats: $activeSamplerResponseSumStats")
    stringBuilder.append(s"\nnumFeaturesStats: $numFeaturesStats")

    stringBuilder.toString()
  }
}

object RandomEffectDataset {

  type ActiveData = RDD[(REId, LocalDataset)]
  type PassiveData = RDD[(UniqueSampleId, (REId, LabeledPoint))]

  /**
   * Build a new [[RandomEffectDataset]] from the raw data using the given configuration.
   *
   * @param gameDataset The RDD of [[GameDatum]] used to generate the random effect dataset
   * @param randomEffectDataConfiguration The data configuration for the random effect dataset
   * @param randomEffectPartitioner The per random effect partitioner used to generated the grouped active data
   * @param existingModelKeysRddOpt Optional set of entities that have existing models
   * @return A new [[RandomEffectDataset]]
   */
  def apply(
      gameDataset: RDD[(UniqueSampleId, GameDatum)],
      randomEffectDataConfiguration: RandomEffectDataConfiguration,
      randomEffectPartitioner: Partitioner,
      existingModelKeysRddOpt: Option[RDD[REId]]): RandomEffectDataset = {

    val activeData = generateActiveData(
      gameDataset,
      randomEffectDataConfiguration,
      randomEffectPartitioner,
      existingModelKeysRddOpt)
    val uniqueIdToRandomEffectIds = generateIdMap(activeData, gameDataset.partitioner.get)
    val passiveData = generatePassiveData(gameDataset, uniqueIdToRandomEffectIds, randomEffectDataConfiguration)

    new RandomEffectDataset(
      activeData,
      passiveData,
      uniqueIdToRandomEffectIds,
      randomEffectDataConfiguration.randomEffectType,
      randomEffectDataConfiguration.featureShardId)
  }

  /**
   * Generate active data.
   *
   * @param gameDataset The input dataset
   * @param randomEffectDataConfiguration The random effect data configuration
   * @param randomEffectPartitioner A random effect partitioner
   * @return The active dataset
   */
  protected[data] def generateActiveData(
      gameDataset: RDD[(UniqueSampleId, GameDatum)],
      randomEffectDataConfiguration: RandomEffectDataConfiguration,
      randomEffectPartitioner: Partitioner,
      existingModelKeysRddOpt: Option[RDD[REId]]): ActiveData = {

    val randomEffectType = randomEffectDataConfiguration.randomEffectType
    val featureShardId = randomEffectDataConfiguration.featureShardId

    // Generate random effect data from raw GAME data for selected shard
    val keyedRandomEffectDataset = gameDataset.map { case (uniqueId, gameData) =>
      val randomEffectId = gameData.idTagToValueMap(randomEffectType)
      val labeledPoint = gameData.generateLabeledPointWithFeatureShardId(featureShardId)

      (randomEffectId, (uniqueId, labeledPoint))
    }

    // Filter data using reservoir sampling if active data size is bounded
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

    val filteredRandomEffectDataset = lowerBoundFilteringOnActiveData(
      groupedRandomEffectDataset,
      randomEffectDataConfiguration.numActiveDataPointsLowerBound,
      existingModelKeysRddOpt)
    val rawActiveData = filteredRandomEffectDataset.mapValues { iterable =>
      LocalDataset(iterable.toArray, isSortedByFirstIndex = false)
    }

    // Filter features if feature dimension of active data is bounded
    featureSelectionOnActiveData(rawActiveData, randomEffectDataConfiguration.numFeaturesToSamplesRatioUpperBound)
  }

  /**
   * Generate a dataset grouped by random effect ID and limited to a maximum number of samples selected via reservoir
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
   * @return An [[RDD]] of data grouped by individual ID
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
   * Filter entities with less data than a given threshold.
   *
   * @param groupedData An [[RDD]] of data grouped by individual ID
   * @param numActiveDataPointsLowerBoundOpt Optional threshold for number of data points require to receive a
   *                                         per-entity model
   * @param existingModelKeysRddOpt Optional set of entities that have existing models
   * @return An [[RDD]] of data grouped by individual ID with IDs that do not meet the minimum threshold removed
   */
  private def lowerBoundFilteringOnActiveData(
      groupedData: RDD[(REId, Iterable[(UniqueSampleId, LabeledPoint)])],
      numActiveDataPointsLowerBoundOpt: Option[Int],
      existingModelKeysRddOpt: Option[RDD[REId]]): RDD[(REId, Iterable[(UniqueSampleId, LabeledPoint)])] =
    numActiveDataPointsLowerBoundOpt
      .map { activeDataLowerBound =>
        existingModelKeysRddOpt match {
          case Some(existingModelKeysRdd) =>
            groupedData.zipPartitions(existingModelKeysRdd, preservesPartitioning = true) { (dataIt, existingKeysIt) =>

              val lookupTable = existingKeysIt.toSet

              dataIt.filter { case (key, data) =>
                (data.size >= activeDataLowerBound) || !lookupTable.contains(key)
              }
            }

          case None =>
            groupedData.filter { case (_, data) =>
              data.size >= activeDataLowerBound
            }
        }
      }
      .getOrElse(groupedData)

  /**
   * Reduce active data feature dimension for entities with few samples. The maximum feature dimension is limited to
   * the number of samples multiplied by the feature dimension ratio. Features are chosen by greatest Pearson
   * correlation score.
   *
   * @param activeData The active dataset
   * @param numFeaturesToSamplesRatioUpperBoundOpt Optional ratio of samples to feature dimension
   * @return The active data with the feature dimension reduced
   */
  private def featureSelectionOnActiveData(
      activeData: RDD[(REId, LocalDataset)],
      numFeaturesToSamplesRatioUpperBoundOpt: Option[Double]): ActiveData =
    numFeaturesToSamplesRatioUpperBoundOpt
      .map { numFeaturesToSamplesRatioUpperBound =>
        activeData.mapValues { localDataset =>

          var numFeaturesToKeep = math.ceil(numFeaturesToSamplesRatioUpperBound * localDataset.numDataPoints).toInt
          // In case the above product overflows
          if (numFeaturesToKeep < 0) numFeaturesToKeep = Int.MaxValue

          localDataset.filterFeaturesByPearsonCorrelationScore(numFeaturesToKeep)
        }
      }
      .getOrElse(activeData)

  /**
   * Generate a map of unique sample id to random effect id for active data samples.
   *
   * @param activeData The active dataset
   * @param partitioner The [[Partitioner]] to use for the [[RDD]] of unique sample ID to random effect ID
   * @return A map of unique sample id to random effect id for active data samples
   */
  protected[data] def generateIdMap(activeData: ActiveData, partitioner: Partitioner): RDD[(UniqueSampleId, REId)] =
    activeData
      .flatMap { case (individualId, localDataset) =>
        localDataset.getUniqueIds.map((_, individualId))
      }
      .partitionBy(partitioner)

  /**
   * Generate passive dataset.
   *
   * @param gameDataset The raw input data
   * @param uniqueIdToRandomEffectIds A map of unique sample id to random effect id for active data samples
   * @param randomEffectDataConfiguration  The random effect data configuration
   * @return The passive dataset
   */
  protected[data] def generatePassiveData(
      gameDataset: RDD[(UniqueSampleId, GameDatum)],
      uniqueIdToRandomEffectIds: RDD[(UniqueSampleId, REId)],
      randomEffectDataConfiguration: RandomEffectDataConfiguration): RDD[(UniqueSampleId, (REId, LabeledPoint))] = {

    val rEType = randomEffectDataConfiguration.randomEffectType
    val shardId = randomEffectDataConfiguration.featureShardId

    // Once the active data is determined, the remaining data becomes passive data
    gameDataset
      .subtractByKey(uniqueIdToRandomEffectIds)
      .mapValues { gameData =>
        val randomEffectId = gameData.idTagToValueMap(rEType)
        val labeledPoint = gameData.generateLabeledPointWithFeatureShardId(shardId)

        (randomEffectId, labeledPoint)
      }
  }
}
