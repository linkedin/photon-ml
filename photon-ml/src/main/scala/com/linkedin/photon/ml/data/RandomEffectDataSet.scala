package com.linkedin.photon.ml.data

import scala.collection.Set
import scala.util.hashing.byteswap64

import org.apache.spark.{SparkContext, Partitioner}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.{StorageLevel => SparkStorageLevel}

import com.linkedin.photon.ml.{BroadcastLike, RDDLike}
import com.linkedin.photon.ml.constants.StorageLevel


/**
 * passiveData + activeData = full sharded data set
 * @param passiveDataOption Flattened data sets used to score the whole data set
 * @param activeData Grouped data sets mostly to train the sharded model and score the whole data set.
 * @author xazhang
 */
class RandomEffectDataSet(
    val activeData: RDD[(String, LocalDataSet)],
    protected[data] val globalIdToIndividualIds: RDD[(Long, String)],
    val passiveDataOption: Option[RDD[(Long, (String, LabeledPoint))]],
    val passiveDataIndividualIdsOption: Option[Broadcast[Set[String]]],
    val randomEffectId: String,
    val featureShardId: String) extends DataSet[RandomEffectDataSet] with RDDLike with BroadcastLike {

  val individualIdPartitioner = activeData.partitioner.get
  val globalIdPartitioner = globalIdToIndividualIds.partitioner.get
  val hasPassiveData = passiveDataOption.isDefined

  override def addScoresToOffsets(scores: KeyValueScore): RandomEffectDataSet = {
    val scoresGroupedByIndividualId = scores.scores.join(globalIdToIndividualIds)
        .map { case (globalId, (score, localId)) => (localId, (globalId, score)) }
        .groupByKey(individualIdPartitioner)
        .mapValues(_.toArray.sortBy(_._1))

    val updatedActiveData = activeData.join(scoresGroupedByIndividualId)
        .mapValues { case (localData, localScore) => localData.addScoresToOffsets(localScore) }

    val updatedPassiveDataOption = passiveDataOption.map { passiveData =>
      passiveData.join(scores.scores)
          .mapValues { case ((individualId, LabeledPoint(response, features, offset, weight)), score) =>
        (individualId, LabeledPoint(response, features, offset + score, weight))
      }
    }

    update(updatedActiveData, updatedPassiveDataOption)
  }

  override def sparkContext: SparkContext = activeData.sparkContext

  override def setName(name: String): this.type = {
    activeData.setName(s"$name: Active data")
    globalIdToIndividualIds.setName(s"$name: Global Id to individual Id")
    passiveDataOption.foreach(_.setName(s"$name: Passive data"))
    this
  }

  override def persistRDD(storageLevel: SparkStorageLevel): this.type = {
    if (!activeData.getStorageLevel.isValid) activeData.persist(storageLevel)
    if (!globalIdToIndividualIds.getStorageLevel.isValid) globalIdToIndividualIds.persist(storageLevel)
    passiveDataOption.foreach { passiveData =>
      if (!passiveData.getStorageLevel.isValid) passiveData.persist(storageLevel)
    }
    this
  }

  override def unpersistRDD(): this.type = {
    if (activeData.getStorageLevel.isValid) activeData.unpersist()
    //TODO: Better way to handle the storage level of globalIdToIndividualIds
//    if (globalIdToIndividualIds.getStorageLevel.isValid) globalIdToIndividualIds.unpersist()
    passiveDataOption.foreach { passiveData =>
      if (passiveData.getStorageLevel.isValid) passiveData.unpersist()
    }
    this
  }

  override def unpersistBroadcast(): this.type = {
    passiveDataIndividualIdsOption.foreach(_.unpersist())
    this
  }

  override def materialize(): this.type = {
    activeData.count()
    globalIdToIndividualIds.count()
    passiveDataOption.foreach(_.count())
    this
  }

  def update(
      updatedActiveData: RDD[(String, LocalDataSet)],
      updatedPassiveDataOption: Option[RDD[(Long, (String, LabeledPoint))]]): RandomEffectDataSet = {
    new RandomEffectDataSet(updatedActiveData, globalIdToIndividualIds, updatedPassiveDataOption,
      passiveDataIndividualIdsOption, randomEffectId, featureShardId)
  }

  override def toSummaryString: String = {
    val numActiveSamples = globalIdToIndividualIds.count()
    val activeSampleWeighSum = activeData.values.map(_.getWeights.map(_._2).sum).sum()
    val activeSampleResponseSum = activeData.values.map(_.getLabels.map(_._2).sum).sum()
    val numPassiveSamples = if (hasPassiveData) passiveDataOption.get.count() else 0
    val passiveSampleResponsesSum = if (hasPassiveData) passiveDataOption.get.values.map(_._2.label).sum() else 0
    val numAllSamples = numActiveSamples + numPassiveSamples
    val numActiveSamplesStats = activeData.values.map(_.numDataPoints).stats()
    val activeSamplerResponseSumStats = activeData.values.map(_.getLabels.map(_._2).sum).stats()
    val numFeaturesStats = activeData.values.map(_.numActiveFeatures).stats()
    val numIdsWithPassiveData =
      if (passiveDataIndividualIdsOption.isDefined) passiveDataIndividualIdsOption.get.value.size else 0
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
   * @param gameDataSet The RDD of [[GameData]] used to generate the random effect data set
   * @param randomEffectDataConfiguration The data configuration for the random effect data set
   * @param randomEffectPartitioner The per random effect partitioner used to generated the grouped active data
   * @return
   */
  def buildWithConfiguration(
      gameDataSet: RDD[(Long, GameData)],
      randomEffectDataConfiguration: RandomEffectDataConfiguration,
      randomEffectPartitioner: Partitioner): RandomEffectDataSet = {

    val randomEffectId = randomEffectDataConfiguration.randomEffectId
    val featureShardId = randomEffectDataConfiguration.featureShardId

    val globalPartitioner = gameDataSet.partitioner.get

    val rawActiveData = generateActiveData(gameDataSet, randomEffectDataConfiguration, randomEffectPartitioner)
    val activeData = featureSelectionOnActiveData(rawActiveData, randomEffectDataConfiguration)
      .setName("Active data")
      .persist(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)

    val globalIdToIndividualIds = activeData.flatMap { case (individualId, localDataSet) =>
      localDataSet.getGlobalIds.map((_, individualId))
    }.partitionBy(globalPartitioner)

    val (passiveDataOption, passiveDataIndividualIdsOption) = if (randomEffectDataConfiguration.isDownSamplingNeeded) {
      val (passiveData, passiveDataIndividualIds) = generatePassiveData(gameDataSet, activeData, globalPartitioner,
        randomEffectDataConfiguration)
      (Option(passiveData), Option(passiveDataIndividualIds))
    } else {
      (None, None)
    }

    new RandomEffectDataSet(activeData, globalIdToIndividualIds, passiveDataOption,
      passiveDataIndividualIdsOption, randomEffectId, featureShardId)
  }

  private def generateActiveData(
      gameDataSet: RDD[(Long, GameData)],
      randomEffectDataConfiguration: RandomEffectDataConfiguration,
      randomEffectPartitioner: Partitioner): RDD[(String, LocalDataSet)] = {

    val randomEffectId = randomEffectDataConfiguration.randomEffectId
    val featureShardId = randomEffectDataConfiguration.featureShardId
    val numActiveDataPointsToKeepUpperBound = randomEffectDataConfiguration.numActiveDataPointsToKeepUpperBound

    val keyedRandomEffectDataSet = gameDataSet.map { case (globalId, gameData) =>
      val individualId = gameData.randomEffectIdToIndividualIdMap(randomEffectId)
      val labeledPoint = gameData.generateLabeledPointWithFeatureShardId(featureShardId)
      (individualId, (globalId, labeledPoint))
    }
    val groupedRandomEffectDataSet =
      if (randomEffectDataConfiguration.isDownSamplingNeeded) {
        val uniqueId = randomEffectId.hashCode
        groupKeyedDataSetViaReservoirSampling(keyedRandomEffectDataSet, randomEffectPartitioner,
          numActiveDataPointsToKeepUpperBound, uniqueId)
      } else {
        keyedRandomEffectDataSet.groupByKey(randomEffectPartitioner)
      }
    groupedRandomEffectDataSet.mapValues(iterable => LocalDataSet(iterable.toArray, isSortedByFirstIndex = false))
  }

  private def groupKeyedDataSetViaReservoirSampling(
      rawKeyedDataSet: RDD[(String, (Long, LabeledPoint))],
      partitioner: Partitioner,
      sampleCap: Int,
      uniqueId: Long): RDD[(String, Iterable[(Long, LabeledPoint)])] = {

    case class ComparableLabeledPointWithId(comparableKey: Int, globalId: Long, labeledPoint: LabeledPoint)
        extends Comparable[ComparableLabeledPointWithId] {

      override def compareTo(comparableLabeledPointWithId: ComparableLabeledPointWithId): Int = {
        if (comparableKey - comparableLabeledPointWithId.comparableKey > 0) 1
        else -1
      }
    }

    val createCombiner =
      (comparableLabeledPointWithId: ComparableLabeledPointWithId) => {
        new MinHeapWithFixedCapacity[ComparableLabeledPointWithId](sampleCap) += comparableLabeledPointWithId
      }
    val mergeValue = (minHeapWithFixedCapacity: MinHeapWithFixedCapacity[ComparableLabeledPointWithId],
                      comparableLabeledPointWithId: ComparableLabeledPointWithId) => {
      minHeapWithFixedCapacity += comparableLabeledPointWithId
    }
    val mergeCombiners = (minHeapWithFixedCapacity1: MinHeapWithFixedCapacity[ComparableLabeledPointWithId],
                          minHeapWithFixedCapacity2: MinHeapWithFixedCapacity[ComparableLabeledPointWithId]) => {
      minHeapWithFixedCapacity1 ++= minHeapWithFixedCapacity2
    }

    /*
    Currently the reservoir sampling algorithm is not fault tolerant, as the comparable key depends on globalId, which
    is compute based on the RDD partition id. The globalId will change after recomputing the RDD after node failure.
    TODO: Need to make sure that the comparableKey is robust to RDD recompute and node failure
     */
    val localDataSets =
      rawKeyedDataSet
        .mapValues { case (globalId, labeledPoint) =>
        val comparableKey = (byteswap64(globalId) ^ byteswap64(uniqueId)).hashCode()
        ComparableLabeledPointWithId(comparableKey, globalId, labeledPoint)
      }
        .combineByKey[MinHeapWithFixedCapacity[ComparableLabeledPointWithId]](createCombiner, mergeValue,
            mergeCombiners, partitioner)
        .mapValues { minHeapWithFixedCapacity =>
        val cumCount = minHeapWithFixedCapacity.cumCount
        val data = minHeapWithFixedCapacity.getData
        val size = data.size
        val weightMultiplierFactor = 1.0 * cumCount / size
        val dataPoints =
          data.map { case ComparableLabeledPointWithId(_, globalId, LabeledPoint(label, features, offset, weight)) =>
            (globalId, LabeledPoint(label, features, offset, weight * weightMultiplierFactor))
          }
        dataPoints
      }
    localDataSets
  }

  private def generatePassiveData(
      gameDataSet: RDD[(Long, GameData)],
      activeData: RDD[(String, LocalDataSet)],
      globalPartitioner: Partitioner,
      randomEffectDataConfiguration: RandomEffectDataConfiguration)
  : (RDD[(Long, (String, LabeledPoint))], Broadcast[Set[String]]) = {

    val randomEffectId = randomEffectDataConfiguration.randomEffectId
    val featureShardId = randomEffectDataConfiguration.featureShardId
    val numPassiveDataPointsToKeepLowerBound = randomEffectDataConfiguration.numPassiveDataPointsToKeepLowerBound

    // The remaining data not included in the active data will be kept as passive data
    val activeDataGlobalIds = activeData.flatMapValues(_.dataPoints.map(_._1)).map(_.swap)
    val keyedRandomEffectDataSet = gameDataSet.mapValues { gameData =>
      val individualId = gameData.randomEffectIdToIndividualIdMap(randomEffectId)
      val labeledPoint = gameData.generateLabeledPointWithFeatureShardId(featureShardId)
      (individualId, labeledPoint)
    }
    val passiveData = keyedRandomEffectDataSet.subtractByKey(activeDataGlobalIds, globalPartitioner)
        .setName("tmp passive data")
        .persist(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)
    val passiveDataIndividualIdCountsMap = passiveData.map { case (_, (individualId, _)) => (individualId, 1) }
        .reduceByKey(_ + _).collectAsMap()
    // Only keep the passive data whose total number of data points is larger than the given lower bound
    val passiveDataIndividualIds = passiveDataIndividualIdCountsMap
        .filter(_._2 > numPassiveDataPointsToKeepLowerBound).keySet
    val sparkContext = gameDataSet.sparkContext
    val passiveDataIndividualIdsBroadcast = sparkContext.broadcast(passiveDataIndividualIds)
    val filteredPassiveData = passiveData.filter { case ((_, (id, _))) =>
      passiveDataIndividualIdsBroadcast.value.contains(id)
    }.setName("passive data").persist(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)
    filteredPassiveData.count()
    passiveData.unpersist()
    (filteredPassiveData, passiveDataIndividualIdsBroadcast)
  }

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
