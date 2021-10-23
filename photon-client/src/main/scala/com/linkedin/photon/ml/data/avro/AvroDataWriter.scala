/*
 * Copyright 2018 LinkedIn Corp. All rights reserved.
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
package com.linkedin.photon.ml.data.avro

import scala.collection.mutable.ListBuffer
import scala.collection.JavaConversions._

import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.hadoop.mapred.JobConf
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

import com.linkedin.photon.avro.generated.{FeatureAvro, SimplifiedResponsePrediction}
import com.linkedin.photon.ml.Constants.DELIMITER
import com.linkedin.photon.ml.index.{IndexMap, IndexMapLoader}

/**
 * Write dataframe to Avro files on HDFS in [[SimplifiedResponsePrediction]] format
 */
class AvroDataWriter {

  import AvroDataWriter._

  private val sparkSession = SparkSession.builder.getOrCreate()
  private val sc = sparkSession.sparkContext

  /**
   * Write the DataFrame into avro records using the given indexMapLoader
   *
   * @param df The DataFrame
   * @param outputPath The output path to store the avro files
   * @param indexMapLoader The IndexMapLoader store feature to index information
   * @param responseColumn The response column name in df
   * @param featureColumn The feature column name in df
   */
  def write(
    df: DataFrame,
    outputPath: String,
    indexMapLoader: IndexMapLoader,
    responseColumn: String,
    featureColumn: String,
    overwrite: Boolean = false): Unit = {

    // TODO: Save other fields in the dataset, i.e. feature columns
    val columns = df.columns
    require(columns.contains(responseColumn), s"There must be a $responseColumn column present in dataframe")
    require(columns.contains(featureColumn), s"There must be a $featureColumn column present in dataframe")

    val hasOffset = columns.contains(OFFSET)
    val hasWeight = columns.contains(WEIGHT)

    val avroDataset = df.rdd.mapPartitions { rows =>
      val indexMap = indexMapLoader.indexMapForRDD()
      val rowBuilder = SimplifiedResponsePrediction.newBuilder()

      rows.map { r: Row =>
        val features = r.getAs[Vector](featureColumn)
        val response = getValueAsDouble(r, responseColumn)
        val offset = if (hasOffset) getValueAsDouble(r, OFFSET) else DEFAULTS(OFFSET)
        val weight = if (hasWeight) getValueAsDouble(r, WEIGHT) else DEFAULTS(WEIGHT)

        rowBuilder
          .setResponse(response)
          .setOffset(offset)
          .setWeight(weight)
          .setFeatures(buildAvroFeatures(features, indexMap))
          .build()
      }
    }

    // Write the converted dataset back to HDFS
    if (overwrite) {
      val output = new Path(outputPath)
      val fs = output.getFileSystem(sc.hadoopConfiguration)
      if (fs.exists(output)) {
        fs.delete(output, true)
      }
    }

    AvroUtils.saveAsAvro[SimplifiedResponsePrediction](
      avroDataset,
      outputPath,
      SimplifiedResponsePrediction.getClassSchema.toString,
      new JobConf(sc.hadoopConfiguration))
  }
}

object AvroDataWriter {

  val OFFSET = "offset"
  val WEIGHT = "weight"
  val DEFAULTS = Map(OFFSET -> 0.0D, WEIGHT -> 1.0D)

  /**
   * Helper function to convert Row index field to double
   *
   * @param row A training record in [[Row]] format
   * @param fieldName The index of particular field
   * @return A double in this field
   */
  protected[data] def getValueAsDouble(row: Row, fieldName: String): Double = {

    row.getAs[Any](fieldName) match {
      case null =>
        DEFAULTS.getOrElse(fieldName, throw new IllegalArgumentException(s"Unsupported null for fieldName $fieldName"))

      case n: Number =>
        n.doubleValue

      case s: String =>
        s.toDouble

      case b: Boolean =>
        if (b) 1.0D else 0.0D

      case _ =>
        throw new IllegalArgumentException(s"Unsupported data type")
    }
  }

  /**
   * Build a list of Avro Feature instances for the given list [[Vector]] and [[IndexMap]]
   *
   * @param vector The extracted feature in [[Vector]] for a particular training instance
   * @param indexMap The reverse index map from feature to index
   * @return A list of Avro Feature instances built from the vector
   */
  protected[data] def buildAvroFeatures(vector: Vector, indexMap: IndexMap): java.util.List[FeatureAvro] = {

    val builder = FeatureAvro.newBuilder()
    val avroFeatures = new ListBuffer[FeatureAvro]
    vector.foreachActive {
      case (vectorIdx, vectorValue) =>
        val feature = indexMap.getFeatureName(vectorIdx).get
        feature.split(DELIMITER) match {
          case Array(name, term) =>
            builder.setName(name).setTerm(term)
          case Array(name) =>
            builder.setName(name).setTerm("")
          case _ =>
            throw new IllegalArgumentException(s"Error parsing the name and term for this feature $feature")
        }
        builder.setValue(vectorValue)
        avroFeatures += builder.build()
    }
    avroFeatures.toList
  }
}
