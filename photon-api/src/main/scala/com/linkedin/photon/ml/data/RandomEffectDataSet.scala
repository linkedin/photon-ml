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

import scala.collection.Set
import scala.util.hashing.byteswap64

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{Partitioner, SparkContext}

import com.linkedin.photon.ml.Types.{FeatureShardId, REId, REType, UniqueSampleId}
import com.linkedin.photon.ml.data.scoring.CoordinateDataScores
import com.linkedin.photon.ml.spark.{BroadcastLike, RDDLike}

/**
 * Data set implementation for random effect datasets:
 *
 *   activeData + passiveData = full sharded data set
 *
 * For the cases where a random effect ID may have a lot of data (enough that it starts causing trouble running on a
 * single node), active and passive data provide tuning options. Active data is the subset of data that is used for
 * both training and scoring (when determining residuals), while passive data is used only for scoring. In the vast
 * majority of cases, all data will be active data.
 *
 * @param activeData Grouped data sets mostly to train the sharded model and score the whole data set.
 * @param uniqueIdToRandomEffectIds Unique id to random effect id map
 * @param passiveDataOption Flattened data sets used to score the whole data set
 * @param passiveDataRandomEffectIdsOption Passive data individual IDs
 * @param randomEffectType The random effect type (e.g. "memberId")
 * @param featureShardId The feature shard ID
 */
protected[ml] class RandomEffectDataSet(
    val activeData: RDD[(REId, LocalDataSet)],
    protected[data] val uniqueIdToRandomEffectIds: RDD[(UniqueSampleId, REId)],
    val passiveDataOption: Option[RDD[(UniqueSampleId, (REId, LabeledPoint))]],
    val passiveDataRandomEffectIdsOption: Option[Broadcast[Set[String]]],
    val randomEffectType: REType,
    val featureShardId: FeatureShardId)
  extends DataSet[RandomEffectDataSet]
  with RDDLike
  with BroadcastLike {

  val randomEffectIdPartitioner: Partitioner = activeData.partitioner.get
  val uniqueIdPartitioner: Partitioner = uniqueIdToRandomEffectIds.partitioner.get
  val hasPassiveData: Boolean = passiveDataOption.isDefined

  /**
   * Add residual scores to the data offsets.
   *
   * @param scores The residual scores
   * @return The dataset with updated offsets
   */
  override def addScoresToOffsets(scores: CoordinateDataScores): RandomEffectDataSet = {

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

  /**
   * Get the Spark context.
   *
   * @return The Spark context
   */
  override def sparkContext: SparkContext = activeData.sparkContext

  /**
   * Assign a given name to [[activeData]], [[uniqueIdToRandomEffectIds]], and [[passiveDataOption]].
   *
   * @note Not used to reference models in the logic of photon-ml, only used for logging currently.
   * @param name The parent name for all [[RDD]]s in this class
   * @return This object with the names [[activeData]], [[uniqueIdToRandomEffectIds]], and [[passiveDataOption]]
   *         assigned
   */
  override def setName(name: String): RandomEffectDataSet = {

    activeData.setName(s"$name: Active data")
    uniqueIdToRandomEffectIds.setName(s"$name: unique id to individual Id")
    passiveDataOption.foreach(_.setName(s"$name: Passive data"))

    this
  }

  /**
   * Set the storage level of [[activeData]], [[uniqueIdToRandomEffectIds]], and [[passiveDataOption]], and persist
   * their values across the cluster the first time they are computed.
   *
   * @param storageLevel The storage level
   * @return This object with the storage level of [[activeData]], [[uniqueIdToRandomEffectIds]], and
   *         [[passiveDataOption]] set
   */
  override def persistRDD(storageLevel: StorageLevel): RandomEffectDataSet = {

    if (!activeData.getStorageLevel.isValid) activeData.persist(storageLevel)
    if (!uniqueIdToRandomEffectIds.getStorageLevel.isValid) uniqueIdToRandomEffectIds.persist(storageLevel)
    passiveDataOption.foreach { passiveData =>
      if (!passiveData.getStorageLevel.isValid) passiveData.persist(storageLevel)
    }

    this
  }

  /**
   * Mark [[activeData]], [[uniqueIdToRandomEffectIds]], and [[passiveDataOption]] as non-persistent, and remove all
   * blocks for them from memory and disk.
   *
   * @return This object with [[activeData]], [[uniqueIdToRandomEffectIds]], and [[passiveDataOption]] marked
   *         non-persistent
   */
  override def unpersistRDD(): RandomEffectDataSet = {

    if (activeData.getStorageLevel.isValid) activeData.unpersist()
    if (uniqueIdToRandomEffectIds.getStorageLevel.isValid) uniqueIdToRandomEffectIds.unpersist()
    passiveDataOption.foreach { passiveData =>
      if (passiveData.getStorageLevel.isValid) passiveData.unpersist()
    }

    this
  }

  /**
   * Materialize [[activeData]], [[uniqueIdToRandomEffectIds]], and [[passiveDataOption]] (Spark [[RDD]]s are lazy
   * evaluated: this method forces them to be evaluated).
   *
   * @return This object with [[activeData]], [[uniqueIdToRandomEffectIds]], and [[passiveDataOption]] materialized
   */
  override def materialize(): RandomEffectDataSet = {

    passiveDataOption match {
      case Some(passiveData) => materializeOnce(activeData, uniqueIdToRandomEffectIds, passiveData)
      case None => materializeOnce(activeData, uniqueIdToRandomEffectIds)
    }

    this
  }

  /**
   * Asynchronously delete cached copies of [[passiveDataRandomEffectIdsOption]] on the executors.
   *
   * @return This object with [[passiveDataRandomEffectIdsOption]] variables unpersisted
   */
  override def unpersistBroadcast(): RandomEffectDataSet = {

    passiveDataRandomEffectIdsOption.foreach(_.unpersist())

    this
  }

  /**
   * Update the dataset.
   *
   * @param updatedActiveData Updated active data
   * @param updatedPassiveDataOption (Optional) Updated passive data
   * @return A new updated dataset
   */
  def update(
      updatedActiveData: RDD[(REId, LocalDataSet)],
      updatedPassiveDataOption: Option[RDD[(UniqueSampleId, (REId, LabeledPoint))]]): RandomEffectDataSet =

    new RandomEffectDataSet(
      updatedActiveData,
      uniqueIdToRandomEffectIds,
      updatedPassiveDataOption,
      passiveDataRandomEffectIdsOption,
      randomEffectType,
      featureShardId)

  /**
   * Build a human-readable summary for [[RandomEffectDataSet]].
   *
   * @return A summary of the object in string representation
   */
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
   * Build the random effect data set with the given configuration.
   *
   * @param gameDataSet The RDD of [[GameDatum]] used to generate the random effect data set
   * @param randomEffectDataConfiguration The data configuration for the random effect data set
   * @param randomEffectPartitioner The per random effect partitioner used to generated the grouped active data
   * @return A new random effect dataset with the given configuration
   */
  protected[ml] def apply(
      gameDataSet: RDD[(UniqueSampleId, GameDatum)],
      randomEffectDataConfiguration: RandomEffectDataConfiguration,
      randomEffectPartitioner: Partitioner,
      existingModelKeysRddOpt: Option[RDD[REId]]): RandomEffectDataSet = {

    val randomEffectType = randomEffectDataConfiguration.randomEffectType
    val featureShardId = randomEffectDataConfiguration.featureShardId

    val gameDataPartitioner = gameDataSet.partitioner.get

    val rawActiveData = generateActiveData(
      gameDataSet,
      randomEffectDataConfiguration,
      randomEffectPartitioner,
      existingModelKeysRddOpt)
    val activeData = featureSelectionOnActiveData(rawActiveData, randomEffectDataConfiguration)
      .setName("Active data")
      .persist(StorageLevel.DISK_ONLY)

    val globalIdToIndividualIds = activeData
      .flatMap { case (individualId, localDataSet) =>
        localDataSet.getUniqueIds.map((_, individualId))
      }
      .partitionBy(gameDataPartitioner)

    val passiveDataOption = randomEffectDataConfiguration
      .numPassiveDataPointsLowerBound
      .map { passiveDataLowerBound =>
        generatePassiveData(
          gameDataSet,
          activeData,
          gameDataPartitioner,
          randomEffectType,
          featureShardId,
          passiveDataLowerBound)
      }

    new RandomEffectDataSet(
      activeData,
      globalIdToIndividualIds,
      passiveDataOption.map(_._1),
      passiveDataOption.map(_._2),
      randomEffectType,
      featureShardId)
  }

  /**
   * Generate active data.
   *
   * @param gameDataSet The input dataset
   * @param randomEffectDataConfiguration The random effect data configuration
   * @param randomEffectPartitioner A random effect partitioner
   * @return The active dataset
   */
  private def generateActiveData(
      gameDataSet: RDD[(UniqueSampleId, GameDatum)],
      randomEffectDataConfiguration: RandomEffectDataConfiguration,
      randomEffectPartitioner: Partitioner,
      existingModelKeysRddOpt: Option[RDD[REId]]): RDD[(REId, LocalDataSet)] = {

    val randomEffectType = randomEffectDataConfiguration.randomEffectType
    val featureShardId = randomEffectDataConfiguration.featureShardId

    val keyedRandomEffectDataSet = gameDataSet.map { case (uniqueId, gameData) =>
      val randomEffectId = gameData.idTagToValueMap(randomEffectType)
      val labeledPoint = gameData.generateLabeledPointWithFeatureShardId(featureShardId)
      (randomEffectId, (uniqueId, labeledPoint))
    }

    val groupedRandomEffectDataSet = randomEffectDataConfiguration
      .numActiveDataPointsUpperBound
      .map { activeDataUpperBound =>
        groupKeyedDataSetViaReservoirSampling(
          keyedRandomEffectDataSet,
          randomEffectPartitioner,
          activeDataUpperBound,
          randomEffectType)
      }
      .getOrElse(keyedRandomEffectDataSet.groupByKey(randomEffectPartitioner))

    randomEffectDataConfiguration
      .numActiveDataPointsLowerBound
      .map { activeDataLowerBound =>
        existingModelKeysRddOpt match {
          case Some(existingModelKeysRdd) =>
            groupedRandomEffectDataSet
              .zipPartitions(existingModelKeysRdd, preservesPartitioning = true) { (dataIt, existingKeysIt) =>

              val lookupTable = existingKeysIt.toSet

              dataIt.filter { case (key, data) =>
                (data.size >= activeDataLowerBound) || !lookupTable.contains(key)
              }
            }

          case None =>
            groupedRandomEffectDataSet.filter { case (_, data) =>
              data.size >= activeDataLowerBound
            }
        }
      }
      .getOrElse(groupedRandomEffectDataSet)
      .mapValues(data => LocalDataSet(data.toArray, isSortedByFirstIndex = false))
  }

  /**
   * Generate a group keyed dataset via reservoir sampling.
   *
   * @param rawKeyedDataSet The raw keyed dataset
   * @param partitioner The partitioner
   * @param sampleCap The sample cap
   * @param randomEffectType The type of random effect
   * @return An RDD of data grouped by individual ID
   */
  private def groupKeyedDataSetViaReservoirSampling(
      rawKeyedDataSet: RDD[(REId, (UniqueSampleId, LabeledPoint))],
      partitioner: Partitioner,
      sampleCap: Int,
      randomEffectType: REType): RDD[(REId, Iterable[(UniqueSampleId, LabeledPoint)])] = {

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
        .combineByKey[MinHeapWithFixedCapacity[ComparableLabeledPointWithId]](
          createCombiner,
          mergeValue,
          mergeCombiners,
          partitioner)
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
   * Generate passive dataset.
   *
   * @param gameDataSet The raw input data set
   * @param activeData The active data set
   * @param gameDataPartitioner A global partitioner
   * @param randomEffectType The corresponding random effect type of the data set
   * @param featureShardId Key of the feature shard used to generate the data set
   * @param passiveDataLowerBound The lower bound on the number of data points required to create a passive data set
   * @return The passive data set
   */
  private def generatePassiveData(
      gameDataSet: RDD[(UniqueSampleId, GameDatum)],
      activeData: RDD[(REId, LocalDataSet)],
      gameDataPartitioner: Partitioner,
      randomEffectType: REType,
      featureShardId: FeatureShardId,
      passiveDataLowerBound: Int):
    (RDD[(UniqueSampleId, (REId, LabeledPoint))], Broadcast[Set[REId]]) = {

    // The remaining data not included in the active data will be kept as passive data
    val activeDataUniqueIds = activeData.flatMapValues(_.dataPoints.map(_._1)).map(_.swap)
    val keyedRandomEffectDataSet = gameDataSet.mapValues { gameData =>
      val randomEffectId = gameData.idTagToValueMap(randomEffectType)
      val labeledPoint = gameData.generateLabeledPointWithFeatureShardId(featureShardId)
      (randomEffectId, labeledPoint)
    }

    val passiveData = keyedRandomEffectDataSet.subtractByKey(activeDataUniqueIds, gameDataPartitioner)
        .setName("tmp passive data")
        .persist(StorageLevel.DISK_ONLY)

    val passiveDataRandomEffectIdCountsMap = passiveData
      .map { case (_, (randomEffectId, _)) =>
        (randomEffectId, 1)
      }
      .reduceByKey(_ + _)
      .collectAsMap()

    // Only keep the passive data whose total number of data points is larger than the given lower bound
    val passiveDataRandomEffectIds = passiveDataRandomEffectIdCountsMap
      .filter(_._2 > passiveDataLowerBound)
      .keySet
    val sparkContext = gameDataSet.sparkContext
    val passiveDataRandomEffectIdsBroadcast = sparkContext.broadcast(passiveDataRandomEffectIds)
    val filteredPassiveData = passiveData
      .filter { case (_, (id, _)) =>
        passiveDataRandomEffectIdsBroadcast.value.contains(id)
      }
      .setName("passive data")
      .persist(StorageLevel.DISK_ONLY)

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
      activeData: RDD[(REId, LocalDataSet)],
      randomEffectDataConfiguration: RandomEffectDataConfiguration): RDD[(REId, LocalDataSet)] = {

    randomEffectDataConfiguration
      .numFeaturesToSamplesRatioUpperBound
      .map { numFeaturesToSamplesRatioUpperBound =>
        activeData.mapValues { localDataSet =>
          var numFeaturesToKeep = math.ceil(numFeaturesToSamplesRatioUpperBound * localDataSet.numDataPoints).toInt

          // In case the above product overflows
          if (numFeaturesToKeep < 0) numFeaturesToKeep = Int.MaxValue
          val filteredLocalDataSet = localDataSet.filterFeaturesByPearsonCorrelationScore(numFeaturesToKeep)

          filteredLocalDataSet
        }
      }
      .getOrElse(activeData)
  }
}
