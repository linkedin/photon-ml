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

import scala.collection.{Map, Set}
import scala.collection.JavaConverters._

import breeze.linalg.Vector
import org.apache.avro.generic.GenericRecord
import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.avro.{AvroFieldNames, AvroUtils}
import com.linkedin.photon.ml.data.GameDatum
import com.linkedin.photon.ml.util.{Utils, VectorUtils}


/**
 * A collection of utility functions on Avro formatted data
 * @author xazhang
 */
object DataProcessingUtils {

  private def getShardIdToFeatureDimensionMap(featureShardIdToFeatureMapMap: Map[String, Map[NameAndTerm, Int]])
  : Map[String, Int] = {

    featureShardIdToFeatureMapMap.map { case (shardId, featureMap) => (shardId, featureMap.values.max + 1) }
  }

  //TODO: Change the scope to protected[avro] after Avro related classes/functIons are decoupled from the rest of code
  protected[ml] def getGameDataSetFromGenericRecords(
      records: RDD[(Long, GenericRecord)],
      featureShardIdToFeatureSectionKeysMap: Map[String, Set[String]],
      featureShardIdToFeatureMapMap: Map[String, Map[NameAndTerm, Int]],
      randomEffectIdSet: Set[String],
      isResponseRequired: Boolean): RDD[(Long, GameDatum)] = {

    val shardIdToFeatureDimensionMap = getShardIdToFeatureDimensionMap(featureShardIdToFeatureMapMap)
    val featureShardIdToFeatureMapMapBroadcast = records.sparkContext.broadcast(featureShardIdToFeatureMapMap)
    records.mapValues(record => getGameDatumFromGenericRecord(
      record,
      featureShardIdToFeatureSectionKeysMap,
      featureShardIdToFeatureMapMapBroadcast.value,
      shardIdToFeatureDimensionMap,
      randomEffectIdSet,
      isResponseRequired
    ))
  }

  protected[ml] def getGameDataSetWithUIDFromGenericRecords(
      records: RDD[(Long, GenericRecord)],
      featureShardIdToFeatureSectionKeysMap: Map[String, Set[String]],
      featureShardIdToFeatureMapMap: Map[String, Map[NameAndTerm, Int]],
      randomEffectIdSet: Set[String],
      isResponseRequired: Boolean): RDD[(Long, (GameDatum, Option[String]))] = {

    val shardIdToFeatureDimensionMap = getShardIdToFeatureDimensionMap(featureShardIdToFeatureMapMap)
    val featureShardIdToFeatureMapMapBroadcast = records.sparkContext.broadcast(featureShardIdToFeatureMapMap)
    records.mapValues { record =>
      val gameDatum = getGameDatumFromGenericRecord(
        record,
        featureShardIdToFeatureSectionKeysMap,
        featureShardIdToFeatureMapMapBroadcast.value,
        shardIdToFeatureDimensionMap,
        randomEffectIdSet,
        isResponseRequired
      )
      val uid = if (record.get(AvroFieldNames.UID) != null) {
        Some(Utils.getStringAvro(record, AvroFieldNames.UID))
      } else {
        None
      }
      (gameDatum, uid)
    }
  }

  private def getGameDatumFromGenericRecord(
      record: GenericRecord,
      featureShardSectionKeys: Map[String, Set[String]],
      featureShardMaps: Map[String, Map[NameAndTerm, Int]],
      shardIdToFeatureDimensionMap: Map[String, Int],
      randomEffectIdSet: Set[String],
      isResponseRequired: Boolean): GameDatum = {

    val featureShardContainer = featureShardSectionKeys.map { case (shardId, featureSectionKeys) =>
      val featureMap = featureShardMaps(shardId)
      val featureDimension = shardIdToFeatureDimensionMap(shardId)
      val features = getFeaturesFromGenericRecord(record, featureMap, featureSectionKeys, featureDimension)
      (shardId, features)
    }
    val response = if (isResponseRequired) {
      Utils.getDoubleAvro(record, AvroFieldNames.RESPONSE)
    } else {
      if (record.get(AvroFieldNames.RESPONSE) != null) {
        Utils.getDoubleAvro(record, AvroFieldNames.RESPONSE)
      } else {
        Double.NaN
      }
    }
    val offset = if (record.get(AvroFieldNames.OFFSET) != null) {
      Utils.getDoubleAvro(record, AvroFieldNames.OFFSET)
    } else {
      0.0
    }

    val weight = if (record.get(AvroFieldNames.WEIGHT) != null) {
      Utils.getDoubleAvro(record, AvroFieldNames.WEIGHT)
    } else {
      1.0
    }

    val ids = randomEffectIdSet.map { randomEffectId =>
      (randomEffectId, Utils.getStringAvro(record, randomEffectId))
    }.toMap
    new GameDatum(response, offset, weight, featureShardContainer, ids)
  }

  private def getFeaturesFromGenericRecord(
      record: GenericRecord,
      nameAndTermToIndexMap: Map[NameAndTerm, Int],
      fieldNames: Set[String],
      featureDimension: Int): Vector[Double] = {

    val featuresAsIndexValueArray = fieldNames.map(fieldName =>
      record.get(fieldName) match {
        case recordList: JList[_] =>
          recordList.asScala.flatMap {
            case record: GenericRecord =>
              val nameAndTerm = AvroUtils.readNameAndTermFromGenericRecord(record)
              if (nameAndTermToIndexMap.contains(nameAndTerm)) {
                Some(nameAndTermToIndexMap(nameAndTerm) -> Utils.getDoubleAvro(record, AvroFieldNames.VALUE))
              } else {
                None
              }
            case any => throw new IllegalArgumentException(s"$any in features list is not a GenericRecord")
          }
        case _ => throw new IllegalArgumentException(s"$fieldName is not a list (or is null).")
      }
    ).foldLeft(Array[(Int, Double)]())(_ ++ _)
    val isAddingInterceptToFeatureMap = nameAndTermToIndexMap.contains(NameAndTerm.INTERCEPT_NAME_AND_TERM)
    if (isAddingInterceptToFeatureMap) {
      VectorUtils.convertIndexAndValuePairArrayToSparseVector(featuresAsIndexValueArray ++
        Array(nameAndTermToIndexMap(NameAndTerm.INTERCEPT_NAME_AND_TERM) -> 1.0), featureDimension)
    } else {
      VectorUtils.convertIndexAndValuePairArrayToSparseVector(featuresAsIndexValueArray, featureDimension)
    }
  }
}
