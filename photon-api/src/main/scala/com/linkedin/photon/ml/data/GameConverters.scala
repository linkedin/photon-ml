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

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.linalg.SparseVector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Row}

import com.linkedin.photon.ml.Types.FeatureShardId
import com.linkedin.photon.ml.util.VectorUtils

/**
 * A collection of utility functions for converting to and from GAME datasets.
 */
object GameConverters {

  // Column name for the synthesized unique id column
  private val UNIQUE_ID_COLUMN_NAME = "___photon:uniqueId___"

  /**
   * Converts a [[DataFrame]] into an [[RDD]] of type [[GameDatum]].
   *
   * @note We "decode" the map of column names into an Array[String] which we broadcast for performance. The
   *       "inputColumnNames" contains the user-specified custom names of columns required by GAME, with default names
   *       for the unspecified columns.
   * @param data The source [[DataFrame]]
   * @param featureShards A set of feature shard ids
   * @param idTagSet The set of columns/metadata fields expected for each [[Row]] in the [[DataFrame]]
   * @param isResponseRequired Whether a response column is mandatory. For example: [[GameDatum]] used for training
   *                           require a response for each [[Row]]; [[GameDatum]] used for scoring do not.
   * @param inputColumnsNames User-supplied input column names to read the input data
   * @return An [[RDD]] of type [[GameDatum]]
   */
  protected[ml] def getGameDataSetFromDataFrame(
      data: DataFrame,
      featureShards: Set[FeatureShardId],
      idTagSet: Set[String],
      isResponseRequired: Boolean,
      inputColumnsNames: InputColumnsNames = InputColumnsNames()): RDD[(Long, GameDatum)] = {

    val recordsWithUniqueId = data.withColumn(UNIQUE_ID_COLUMN_NAME, monotonically_increasing_id())
    val inputColumnsNamesBroadcast = data.sqlContext.sparkContext.broadcast(inputColumnsNames)

    recordsWithUniqueId
      .rdd
      .map { row: Row =>
        (row.getAs[Long](UNIQUE_ID_COLUMN_NAME),
          getGameDatumFromRow(row, featureShards, idTagSet, isResponseRequired, inputColumnsNamesBroadcast))
      }
  }

  /**
   * Build a [[GameDatum]] from a [[DataFrame]] [[Row]].
   *
   * @param row The source [[DataFrame]] [[Row]] (must contain [[SparseVector]] instances)
   * @param featureShards A set of feature shard ids
   * @param idTagSet The set of columns/metadata fields expected for the [[Row]]
   * @param isResponseRequired Whether a response column is mandatory. For example: [[GameDatum]] used for training
   *                           require a response for the [[Row]]; [[GameDatum]] used for scoring do not.
   * @param columnsBroadcast The names of the columns to look for in the input rows, in order
   * @return A [[GameDatum]]
   */
  protected[data] def getGameDatumFromRow(
      row: Row,
      featureShards: Set[String],
      idTagSet: Set[String],
      isResponseRequired: Boolean,
      columnsBroadcast: Broadcast[InputColumnsNames]): GameDatum = {

    val columns = columnsBroadcast.value

    val featureShardContainer = featureShards.map { shardId =>
      val features = row.getAs[SparseVector](shardId)
      (shardId, VectorUtils.mllibToBreeze(features))
    }.toMap

    val response = if (isResponseRequired) {
      row.getAs[Number](columns(InputColumnsNames.RESPONSE)).doubleValue
    } else {
      if (row.schema.fieldNames.contains(columns(InputColumnsNames.RESPONSE))) {
        row.getAs[Number](columns(InputColumnsNames.RESPONSE)).doubleValue
      } else {
        Double.NaN
      }
    }

    val offset = if (row.schema.fieldNames.contains(columns(InputColumnsNames.OFFSET))) {
      Option(row.getAs[Number](columns(InputColumnsNames.OFFSET))).map(_.doubleValue)
    } else {
      None
    }

    val weight = if (row.schema.fieldNames.contains(columns(InputColumnsNames.WEIGHT))) {
      Option(row.getAs[Number](columns(InputColumnsNames.WEIGHT))).map(_.doubleValue)
    } else {
      None
    }

    val idTagToValueMap =
      // TODO: find a better way to handle the field "uid", which is used in ScoringResult
      if (row.schema.fieldNames.contains(columns(InputColumnsNames.UID))
          && row.getAs[Any](columns(InputColumnsNames.UID)) != null) {
        getIdTagToValueMapFromRow(row, idTagSet, columns) +
            (InputColumnsNames.UID.toString -> row.getAs[Any](columns(InputColumnsNames.UID)).toString)
      } else {
        getIdTagToValueMapFromRow(row, idTagSet, columns)
      }

    new GameDatum(
      response,
      offset,
      weight,
      featureShardContainer,
      idTagToValueMap)
  }

  /**
   * Given a [[DataFrame]] [[Row]], build a map of ID tag to ID value.
   *
   * @param row The source DataFrame row
   * @param idTagSet The set of columns/metadata fields expected for the [[Row]]
   * @return The map of ID tag to ID value map for the [[Row]]
   */
  protected[data] def getIdTagToValueMapFromRow(
    row: Row,
    idTagSet: Set[String],
    columns: InputColumnsNames = InputColumnsNames()): Map[String, String] = {

    val metaMap: Option[Map[String, String]] = if (row.schema.fieldNames.contains(columns(InputColumnsNames.META_DATA_MAP))) {
      Some(row.getAs[Map[String, String]](columns(InputColumnsNames.META_DATA_MAP)))
    } else {
      None
    }

    idTagSet
      .map { idTag =>
        val idFromRow: Option[String] = if (row.schema.fieldNames.contains(idTag)) {
          Some(row.getAs[Any](idTag).toString)
        } else {
          None
        }

        val id = idFromRow
          .orElse {
            metaMap.flatMap(_.get(idTag))
          }
          .getOrElse(
            throw new IllegalArgumentException(
              s"Cannot find id in either record field: $idTag or in metadataMap with key: #$idTag"))

        // random effect group name -> random effect group id value
        // random effect types are assumed to be strings
        (idTag, id)
      }
      .toMap
  }
}
