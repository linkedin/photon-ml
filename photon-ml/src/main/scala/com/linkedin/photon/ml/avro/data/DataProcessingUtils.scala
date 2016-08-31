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

import breeze.linalg.Vector
import com.linkedin.photon.ml.avro.{AvroFieldNames, AvroUtils}
import com.linkedin.photon.ml.data.GameDatum
import com.linkedin.photon.ml.io.GLMSuite
import com.linkedin.photon.ml.util.{IndexMap, IndexMapLoader, Utils, VectorUtils}
import org.apache.avro.generic.GenericRecord
import org.apache.spark.rdd.RDD

import scala.collection.JavaConverters._
import scala.collection.{Map, Set}

/**
  * A collection of utility functions on Avro formatted data
  */
object DataProcessingUtils {

  private def getShardIdToFeatureDimensionMap(
      featureShardIdToFeatureMapLoader: Map[String, IndexMapLoader]): Map[String, Int] = {

    featureShardIdToFeatureMapLoader.map { case (shardId, featureMapLoader) =>
      (shardId, featureMapLoader.indexMapForDriver.featureDimension)
    }
  }

  //TODO: Change the scope to protected[avro] after Avro related classes/functIons are decoupled from the rest of code
  protected[ml] def getGameDataSetFromGenericRecords(
      records: RDD[(Long, GenericRecord)],
      featureShardIdToFeatureSectionKeysMap: Map[String, Set[String]],
      featureShardIdToFeatureMapLoader: Map[String, IndexMapLoader],
      randomEffectIdSet: Set[String],
      isResponseRequired: Boolean): RDD[(Long, GameDatum)] = {

    val shardIdToFeatureDimensionMap = getShardIdToFeatureDimensionMap(featureShardIdToFeatureMapLoader)

    records.mapPartitions { iter =>
      val featureShardIdToFeatureMap = featureShardIdToFeatureMapLoader.map { case (shardId, loader) =>
        (shardId, loader.indexMapForRDD)
      }.toMap

      iter.map { case (id, record) => (id, getGameDatumFromGenericRecord(
        record,
        featureShardIdToFeatureSectionKeysMap,
        featureShardIdToFeatureMap,
        shardIdToFeatureDimensionMap,
        randomEffectIdSet,
        isResponseRequired
      ))}
    }
  }

  protected[ml] def getGameDataSetWithUIDFromGenericRecords(
      records: RDD[(Long, GenericRecord)],
      featureShardIdToFeatureSectionKeysMap: Map[String, Set[String]],
      featureShardIdToFeatureMapLoader: Map[String, IndexMapLoader],
      randomEffectIdSet: Set[String],
      isResponseRequired: Boolean): RDD[(Long, (GameDatum, Option[String]))] = {

    val shardIdToFeatureDimensionMap = getShardIdToFeatureDimensionMap(featureShardIdToFeatureMapLoader)

    records.mapPartitions { iter =>
      val featureShardIdToFeatureMap = featureShardIdToFeatureMapLoader.map { case (shardId, loader) =>
        (shardId, loader.indexMapForRDD)
      }.toMap

      iter.map { case (id, record) =>
        val gameDatum = getGameDatumFromGenericRecord(
          record,
          featureShardIdToFeatureSectionKeysMap,
          featureShardIdToFeatureMap,
          shardIdToFeatureDimensionMap,
          randomEffectIdSet,
          isResponseRequired
        )

        val uid = if (record.get(AvroFieldNames.UID) != null) {
          Some(Utils.getStringAvro(record, AvroFieldNames.UID))
        } else {
          None
        }

        (id, (gameDatum, uid))
      }
    }
  }

  /**
    * Given a GenericRecord, build the random effect id map:
    *     (random effect name -> random effect id value)
    *
    * @note Exposed for testing purpose.
    * @param record the avro generic record
    * @param randomEffectIdSet a set of random effect id names
    * @return the random effect id map of (name -> value)
    */
  protected[avro] def makeRandomEffectIdMap(
      record: GenericRecord,
      randomEffectIdSet: Set[String]): Map[String, String] = {

    val metaMap = Utils.getMapAvro(record, AvroFieldNames.META_DATA_MAP, isNullOK = true)

    randomEffectIdSet.map { randomEffectId =>
      val idValue = Utils.getStringAvro(record, randomEffectId, isNullOK = true)

      val finalIdValue = if (idValue.isEmpty) {
        val mapIdValue = if (metaMap != null) metaMap.get(randomEffectId) else null
        if (mapIdValue == null) {
          throw new IllegalArgumentException(s"Cannot find id in either record" +
            s"field: $randomEffectId or in metadataMap with key: #$randomEffectId")
        }
        mapIdValue
      } else {
        idValue
      }

      // random effect group name -> random effect group id value
      // random effect ids are assumed to be strings
      (randomEffectId, finalIdValue.toString)
    }.toMap
  }

  private def getGameDatumFromGenericRecord(
      record: GenericRecord,
      featureShardSectionKeys: Map[String, Set[String]],
      featureShardMaps: Map[String, IndexMap],
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

    new GameDatum(response, offset, weight, featureShardContainer,
        makeRandomEffectIdMap(record, randomEffectIdSet))
  }

  private def getFeaturesFromGenericRecord(
      record: GenericRecord,
      featureMap: IndexMap,
      fieldNames: Set[String],
      featureDimension: Int): Vector[Double] = {

    val featuresAsIndexValueArray = fieldNames.map(fieldName =>
      record.get(fieldName) match {
        case recordList: JList[_] =>
          recordList.asScala.flatMap {
            case record: GenericRecord =>
              val nameAndTerm = AvroUtils.readNameAndTermFromGenericRecord(record)
              val featureKey = Utils.getFeatureKey(nameAndTerm.name, nameAndTerm.term)
              if (featureMap.contains(featureKey)) {
                Some(featureMap.getIndex(featureKey) -> Utils.getDoubleAvro(record, AvroFieldNames.VALUE))
              } else {
                None
              }
            case any => throw new IllegalArgumentException(s"$any in features list is not a GenericRecord")
          }
        case _ => throw new IllegalArgumentException(s"$fieldName is not a list (or is null).")
      }
    ).foldLeft(Array[(Int, Double)]())(_ ++ _)
    val isAddingInterceptToFeatureMap = featureMap.contains(GLMSuite.INTERCEPT_NAME_TERM)
    if (isAddingInterceptToFeatureMap) {
      VectorUtils.convertIndexAndValuePairArrayToSparseVector(featuresAsIndexValueArray ++
        Array(featureMap.getIndex(GLMSuite.INTERCEPT_NAME_TERM) -> 1.0), featureDimension)
    } else {
      VectorUtils.convertIndexAndValuePairArrayToSparseVector(featuresAsIndexValueArray, featureDimension)
    }
  }
}
