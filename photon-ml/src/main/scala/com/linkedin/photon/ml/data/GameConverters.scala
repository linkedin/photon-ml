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

import org.apache.spark.mllib.linalg.SparseVector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Row}

import com.linkedin.photon.ml.util.VectorUtils

/**
  * A collection of utility functions for converting to and from GAME datasets
  */
object GameConverters {

  /**
   * Standard field names
   */
  object FieldNames {
    val NAME: String = "name"
    val TERM: String = "term"
    val VALUE: String = "value"
    val RESPONSE: String = "response"
    val OFFSET: String = "offset"
    val WEIGHT: String = "weight"
    val UID: String = "uid"
    val META_DATA_MAP: String = "metadataMap"
  }

  // Column name for the synthesized unique id column
  val UNIQUE_ID_COLUMN_NAME = "___photon:uniqueId___"

  /**
   * Converts a DataFrame into an [[RDD]] of type [[GameDatum]]
   *
   * @param data the source DataFrame
   * @param featureShards a set of feature shard ids
   * @param idTypeSet a set of id types expected to be found in the row
   * @param isResponseRequired whether the response variable is expected to be found in the row. For example, if GAME
   *   data set to be parsed is used for model training, then the response variable is expected to be found in row. If
   *   the GAME data set is used for scoring, then we don't expect to find response.
   * @return the [[RDD]] of type [[GameDatum]]
   */
  protected[ml] def getGameDataSetFromDataFrame(
    data: DataFrame,
    featureShards: Set[String],
    idTypeSet: Set[String],
    isResponseRequired: Boolean): RDD[(Long, GameDatum)] = {

    // Add unique id
    val recordsWithUniqueId = data.withColumn(UNIQUE_ID_COLUMN_NAME, monotonicallyIncreasingId)

    recordsWithUniqueId.rdd.map { row: Row =>
      val id = row.getAs[Long](UNIQUE_ID_COLUMN_NAME)
      (id, getGameDatumFromRow(
        row,
        featureShards,
        idTypeSet,
        isResponseRequired
      ))
    }
  }

  /**
   * Given a DataFrame row, build the id type to value map: (id type -> id value)
   *
   * @param row the source DataFrame row
   * @param idTypeSet the id types to look for from the row, either at the top layer or within "metadataMap"
   * @return the id type to value map in the form of (id type -> id value)
   */
  protected[data] def getIdTypeToValueMapFromRow(
      row: Row,
      idTypeSet: Set[String]): Map[String, String] = {

    val metaMap = if (row.schema.fieldNames.contains(FieldNames.META_DATA_MAP)) {
      Some(row.getAs[Map[String, String]](FieldNames.META_DATA_MAP))
    } else {
      None
    }

    idTypeSet.map { idType =>
      val idFromRow = if (row.schema.fieldNames.contains(idType)) {
        Some(row.getAs[Any](idType).toString)
      } else {
        None
      }

      val id = idFromRow.orElse {
        metaMap.flatMap(_.get(idType))
      }.getOrElse(throw new IllegalArgumentException(s"Cannot find id in either record" +
        s"field: $idType or in metadataMap with key: #$idType"))

      // random effect group name -> random effect group id value
      // random effect types are assumed to be strings
      (idType, id)
    }.toMap
  }

  /**
   * Build a [[GameDatum]] from a DataFrame row
   *
   * @param row the source DataFrame row
   * @param featureShards a set of feature shard ids
   * @param idTypeSet a set of id types expected to be found in the row
   * @param isResponseRequired whether the response variable is expected to be found in the row. For example, if GAME
   *   data set to be parsed is used for model training, then the response variable is expected to be found in row. If
   *   the GAME data set is used for scoring, then we don't expect to find response.
   * @return the [[GameDatum]]
   */
  protected[data] def getGameDatumFromRow(
      row: Row,
      featureShards: Set[String],
      idTypeSet: Set[String],
      isResponseRequired: Boolean): GameDatum = {

    val featureShardContainer = featureShards.map { shardId =>
      val features = row.getAs[SparseVector](shardId)
      (shardId, VectorUtils.mllibToBreeze(features))
    }.toMap

    val response = if (isResponseRequired) {
      row.getAs[Number](FieldNames.RESPONSE).doubleValue
    } else {
      if (row.schema.fieldNames.contains(FieldNames.RESPONSE)) {
        row.getAs[Number](FieldNames.RESPONSE).doubleValue
      } else {
        Double.NaN
      }
    }

    val offset = if (row.schema.fieldNames.contains(FieldNames.OFFSET)) {
      Option(row.getAs[Number](FieldNames.OFFSET)).map(_.doubleValue)
    } else {
      None
    }

    val weight = if (row.schema.fieldNames.contains(FieldNames.WEIGHT)) {
      Option(row.getAs[Number](FieldNames.WEIGHT)).map(_.doubleValue)
    } else {
      None
    }

    val idTypeToValueMap =
      //TODO: find a better way to handle the field "uid", which is used in ScoringResult
      if (row.schema.fieldNames.contains(FieldNames.UID) && row.getAs[Any](FieldNames.UID) != null) {
        getIdTypeToValueMapFromRow(row, idTypeSet) +
            (FieldNames.UID -> row.getAs[Any](FieldNames.UID).toString)
      } else {
        getIdTypeToValueMapFromRow(row, idTypeSet)
      }

    new GameDatum(
      response,
      offset,
      weight,
      featureShardContainer,
      idTypeToValueMap)
  }
}
