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
package com.linkedin.photon.ml.avro.data

import java.util.{List => JList}

import scala.collection.{Map, Set, mutable}

import breeze.linalg.Vector
import org.apache.avro.generic.GenericRecord
import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.avro.{AvroFieldNames, AvroUtils}
import com.linkedin.photon.ml.data.GameData
import com.linkedin.photon.ml.util.{Utils, VectorUtils}


/**
 * A collection of utility functions on Avro formatted data
 * @author xazhang
 */
object DataProcessingUtils {

  //TODO: Change the scope to [[com.linkedin.photon.ml.avro]] after Avro related classes/functons are decoupled from the rest of code
  protected[ml] def parseAndGenerateGameDataSetFromGenericRecords(
      records: RDD[GenericRecord],
      featureShardIdToFeatureSectionKeysMap: Map[String, Set[String]],
      featureShardIdToFeatureMapMap: Map[String, Map[NameAndTerm, Int]],
      randomEffectIdSet: Set[String]): RDD[(Long, GameData)] = {

    val sparkContext = records.sparkContext
    val numPartitions = records.partitions.length.toLong
    val featureShardMapLargestIndices =
      featureShardIdToFeatureMapMap.map { case (shardId, featureMap) =>
      (shardId, featureMap.values.max)
    }
    val featureShardIdToFeatureMapMapBroadcast = sparkContext.broadcast(featureShardIdToFeatureMapMap)
    records.mapPartitionsWithIndex { case (k, iterator) =>
      val featureShardIdToFeatureMapMap = featureShardIdToFeatureMapMapBroadcast.value
      iterator.zipWithIndex.map { case (record, idx) =>
        val gameData = parseAndGenerateGameDataFromGenericRecord(record, featureShardIdToFeatureSectionKeysMap,
          featureShardIdToFeatureMapMap, featureShardMapLargestIndices, randomEffectIdSet)
        (idx * numPartitions + k, gameData)
      }
    }
  }

  private def parseAndGenerateGameDataFromGenericRecord(
      record: GenericRecord,
      featureShardSectionKeys: Map[String, Set[String]],
      featureShardMaps: Map[String, Map[NameAndTerm, Int]],
      featureShardMapLargestIndices: Map[String, Int],
      randomEffectIdSet: Set[String]): GameData = {

    val featureShardContainer = featureShardSectionKeys.map { case (shardId, featureSectionKeys) =>
      val featureMap = featureShardMaps(shardId)
      val largestIndex = featureShardMapLargestIndices(shardId)
      val features = parseAndGenerateFeaturesFromGenericRecord(record, featureMap, featureSectionKeys, largestIndex)
      (shardId, features)
    }
    val response = Utils.getDoubleAvro(record, AvroFieldNames.RESPONSE)
    val offset =
      if (record.get(AvroFieldNames.OFFSET) != null) Utils.getDoubleAvro(record, AvroFieldNames.OFFSET)
      else 0.0
    val weight =
      if (record.get(AvroFieldNames.WEIGHT) != null) Utils.getDoubleAvro(record, AvroFieldNames.WEIGHT)
      else 1.0
    val ids = randomEffectIdSet.map { randomEffectId =>
      (randomEffectId, Utils.getStringAvro(record, randomEffectId, isNullOK = false))
    }.toMap
    new GameData(response, offset, weight, featureShardContainer, ids)
  }

  private def parseAndGenerateFeaturesFromGenericRecord(
      record: GenericRecord,
      featureNameAndTermToIndexMap: Map[NameAndTerm, Int],
      fieldNames: Set[String],
      maxFeatureIndex: Int): Vector[Double] = {

    val featuresAsIndexValueArray = fieldNames.map(fieldName =>
      record.get(fieldName) match {
        case recordList: JList[_] =>
          val featureIndexAndValueArrayBuilder = new mutable.ArrayBuffer[(Int, Double)]
          val iterator = recordList.iterator
          while (iterator.hasNext) {
            iterator.next match {
              case record: GenericRecord =>
                val featureNameAndTerm = AvroUtils.getNameAndTermFromAvroRecord(record)
                if (featureNameAndTermToIndexMap.contains(featureNameAndTerm)) {
                  featureIndexAndValueArrayBuilder += featureNameAndTermToIndexMap(featureNameAndTerm) ->
                      Utils.getDoubleAvro(record, AvroFieldNames.VALUE)
                }
              case any => throw new IllegalArgumentException(s"$any in features list is not a GenericRecord")
            }
          }
          // Dedup the features with the same name by summing up the values
          featureIndexAndValueArrayBuilder
              .groupBy(_._1)
              .map { case (index, values) => (index, values.map(_._2).sum) }
              .toArray
        case _ => throw new IllegalArgumentException(s"$fieldName is not a list (or is null).")
      }
    ).foldLeft(Array[(Int, Double)]())(_ ++ _)
    val isAddingInterceptToFeatureMap = featureNameAndTermToIndexMap.contains(NameAndTerm.INTERCEPT_NAME_AND_TERM)
    if (isAddingInterceptToFeatureMap) {
      VectorUtils.convertIndexAndValuePairArrayToSparseVector(featuresAsIndexValueArray ++
          Array(featureNameAndTermToIndexMap(NameAndTerm.INTERCEPT_NAME_AND_TERM) -> 1.0), maxFeatureIndex + 1)
    } else {
      VectorUtils.convertIndexAndValuePairArrayToSparseVector(featuresAsIndexValueArray, maxFeatureIndex + 1)
    }
  }
}
