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
      (shardId, featureMapLoader.indexMapForDriver().featureDimension)
    }
  }

  /**
   * Parse a [[RDD]] of type [[GameDatum]] from a [[RDD]] of type [[GenericRecord]]
   *
   * @param records a [[RDD]] of type [[GenericRecord]]
   * @param featureShardIdToFeatureSectionKeysMap a map from feature shard id (defined by the user) to feature
   *                                              section keys (defined in the input data's Avro schema)
   * @param featureShardIdToIndexMapLoader a map from feature shard id (defined by the user) to the index map loader
   *                                       [[IndexMapLoader]].
   * @param idTypeSet a set of id types expected to be found and parsed in the Avro records
   * @param isResponseRequired whether the response variable is expected to be found in the Avro records. For example,
   *                           if GAME data set to be parsed is used for model training, then the response variable is
   *                           expected to be found from the Avro records. If the GAME data set is used for scoring,
   *                           then we don't expect to find response.
   * @todo Change the scope to protected[avro] after Avro related classes/functions are decoupled from the rest of code
   * @return parsed [[RDD]] of type [[GameDatum]]
   */
  protected[ml] def getGameDataSetFromGenericRecords(
      records: RDD[(Long, GenericRecord)],
      featureShardIdToFeatureSectionKeysMap: Map[String, Set[String]],
      featureShardIdToIndexMapLoader: Map[String, IndexMapLoader],
      idTypeSet: Set[String],
      isResponseRequired: Boolean): RDD[(Long, GameDatum)] = {

    val shardIdToFeatureDimensionMap = getShardIdToFeatureDimensionMap(featureShardIdToIndexMapLoader)

    records.mapPartitions { iter =>
      val featureShardIdToIndexMap = featureShardIdToIndexMapLoader.map { case (shardId, loader) =>
        (shardId, loader.indexMapForRDD())
      }.toMap

      iter.map { case (id, record) => (id, getGameDatumFromGenericRecord(
        record,
        featureShardIdToFeatureSectionKeysMap,
        featureShardIdToIndexMap,
        shardIdToFeatureDimensionMap,
        idTypeSet,
        isResponseRequired
      ))}
    }
  }

  /**
   * Given a GenericRecord, build the id type to value map: (id type -> id value)
   *
   * @note Exposed for testing purpose.
   * @param record the avro generic record
   * @param idTypeSet the id types to look for from the generic record, either at the top layer or within "metadataMap"
   * @return the id type to value map in the form of (id type -> id value)
   */
  protected[avro] def getIdTypeToValueMapFromGenericRecord(
      record: GenericRecord,
      idTypeSet: Set[String]): Map[String, String] = {

    val metaMap = Utils.getMapAvro(record, AvroFieldNames.META_DATA_MAP, isNullOK = true)

    idTypeSet.map { idType =>
      val idValue = Utils.getStringAvro(record, idType, isNullOK = true)

      val finalIdValue = if (idValue.isEmpty) {
        val mapIdValue = if (metaMap != null) metaMap.get(idType) else null
        if (mapIdValue == null) {
          throw new IllegalArgumentException(s"Cannot find id in either record" +
            s"field: $idType or in metadataMap with key: #$idType")
        }
        mapIdValue
      } else {
        idValue
      }

      // random effect group name -> random effect group id value
      // random effect ids are assumed to be strings
      (idType, finalIdValue.toString)
    }.toMap
  }

  /**
   * Parse a [[GameDatum]] from a [[GenericRecord]]
   * @param record an instance of [[GenericRecord]]
   * @param featureShardIdToFeatureSectionKeysMap a map from feature shard id (defined by the user) to feature
   *                                              section keys (defined in the input data's Avro schema)
   * @param featureShardIdToIndexMap a map from feature shard id (defined by the user) to that feature shard's index map
   *                                    [[IndexMap]] (loaded by the [[IndexMapLoader]])
   * @param idTypeSet a set of id types expected to be found and parsed in the Avro records
   * @param shardIdToFeatureDimensionMap a map from shard Id to that feature shard's dimension
   * @param isResponseRequired whether the response variable is expected to be found in the Avro records. For example,
   *                           if GAME data set to be parsed is used for model training, then the response variable is
   *                           expected to be found from the Avro records. If the GAME data set is used for scoring,
   *                           then we don't expect to find response.
   * @todo Change the scope to protected[avro] after Avro related classes/functions are decoupled from the rest of code
   * @return parsed [[GameDatum]]
   */
  private def getGameDatumFromGenericRecord(
      record: GenericRecord,
      featureShardIdToFeatureSectionKeysMap: Map[String, Set[String]],
      featureShardIdToIndexMap: Map[String, IndexMap],
      shardIdToFeatureDimensionMap: Map[String, Int],
      idTypeSet: Set[String],
      isResponseRequired: Boolean): GameDatum = {

    val featureShardContainer = featureShardIdToFeatureSectionKeysMap.map { case (shardId, featureSectionKeys) =>
      val featureMap = featureShardIdToIndexMap(shardId)
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
      Some(Utils.getDoubleAvro(record, AvroFieldNames.OFFSET))
    } else {
      None
    }
    val weight = if (record.get(AvroFieldNames.WEIGHT) != null) {
      Some(Utils.getDoubleAvro(record, AvroFieldNames.WEIGHT))
    } else {
      None
    }

    new GameDatum(response, offset, weight, featureShardContainer,
        getIdTypeToValueMapFromGenericRecord(record, idTypeSet))
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
