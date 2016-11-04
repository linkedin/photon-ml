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

import com.linkedin.photon.ml.util.{DefaultIndexMapLoader, IndexMap, IndexMapLoader, Utils}

import com.databricks.spark.avro._
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.{SparseVector, VectorUDT}
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.sql.functions.{col, array, udf}
import org.apache.spark.sql.types.{StructField, StructType}

/**
 * A DataReader implementation that reads the Avro format
 *
 * @param sc the Spark context
 * @param defaultFeatureColumn the default column to use for features
 */
class AvroDataReader(
    sc: SparkContext,
    defaultFeatureColumn: String = "features")
  extends DataReader(defaultFeatureColumn) {

  import AvroDataReader._

  /**
   * Internal sql context
   */
  private val sqlContext = new SQLContext(sc)

  /**
   * Reads the avro file at the given path into a DataFrame, generating a default index map for feature names. Merges
   * source columns into combined feature vectors as specified by the featureColumnMap argument. Often features are
   * joined from different sources, and it can be more scalable to combine them into problem-specific feature vectors
   * that can be independently distributed.
   *
   * @param path the path to the avro file or folder
   * @param featureColumnMap a map that specifies how the feature columns should be merged. The keys specify the name
   *   of the merged destination column, and the values are sets of source columns to merge, e.g.:
   *
   *     Map("userFeatures" -> Set("profileFeatures", "titleFeatures"))
   *
   *   This configuration merges the "profileFeatures" and "titleFeatures" columns into a single column named
   *   "userFeatures".
   * @return the loaded and transformed DataFrame
   */
  override def readMerged(
      path: String,
      featureColumnMap: Map[String, Set[String]]): (DataFrame, Map[String, IndexMapLoader]) = {

    val data = sqlContext.read.avro(path)
    val merged = mergeColumns(data, featureColumnMap)
    val indexMapLoaders = generateIndexMapLoaders(merged, featureColumnMap.keys.toSet)

    val transformed = vectorize(merged, indexMapLoaders)

    (transformed, indexMapLoaders)
  }

  /**
   * Reads the avro file at the given path into a DataFrame, using the given index map for feature names. Merges source
   * columns into combined feature vectors as specified by the featureColumnMap argument. Often features are joined from
   * different sources, and it can be more scalable to combine them into problem-specific feature vectors that can be
   * independently distributed.
   *
   * @param path the path to the avro file or folder
   * @param indexMapLoaders a map of index map loaders, containing one loader for each merged feature column
   * @param featureColumnMap a map that specifies how the feature columns should be merged. The keys specify the name
   *   of the merged destination column, and the values are sets of source columns to merge, e.g.:
   *
   *     Map("userFeatures" -> Set("profileFeatures", "titleFeatures"))
   *
   *   This configuration merges the "profileFeatures" and "titleFeatures" columns into a single column named
   *   "userFeatures".
   * @return the loaded and transformed DataFrame
   */
  override def readMerged(
      path: String,
      indexMapLoaders: Map[String, IndexMapLoader],
      featureColumnMap: Map[String, Set[String]]): DataFrame = {

    val data = sqlContext.read.avro(path)

    vectorize(
      mergeColumns(data, featureColumnMap),
      indexMapLoaders)
  }

  /**
   * Transforms the feature column consisting of name, term, and value feature tuples into a column of feature vectors
   *
   * @param df the source DataFrame
   * @param columnName the feature column to convert into a vector
   * @param indexMapLoader the index map loader for this column
   * @return the transformed DataFrame
   */
  protected def vectorize(
      df: DataFrame,
      columnName: String,
      indexMapLoader: IndexMapLoader): DataFrame = {

    val vectorized = df.mapPartitions { rows =>
      val featureMap = indexMapLoader.indexMapForRDD()

      rows.map { row: Row =>
        val features = row.getAs[Seq[Row]](columnName).flatMap {
          case Row(featureKey: String, value: Double) =>
            Some(featureKey -> value)
          case _ => None
        }

        // Check for duplicate features
        val duplicateFeatures = features
          .groupBy(_._1)
          .filter(_._2.length > 1)
        require(duplicateFeatures.isEmpty, s"Duplicate features found: ${duplicateFeatures.toString}")

        // Use the feature map to convert feature names to indices
        val featuresWithIndices = features.flatMap { case (key, value) =>
          if (featureMap.contains(key)) {
            Some(featureMap.getIndex(key) -> value)
          } else {
            None
          }
        }

        // Add intercept if it exists in the feature map
        val featuresWithIntercept = if (featureMap.contains(AvroDataReader.INTERCEPT_KEY)) {
          featuresWithIndices :+ (featureMap.getIndex(AvroDataReader.INTERCEPT_KEY) -> 1.0)
        } else {
          featuresWithIndices
        }

        // Build the vector
        val (indices, values) = featuresWithIntercept.sorted.unzip
        val vector = new SparseVector(featureMap.featureDimension, indices.toArray, values.toArray)

        // Add the column
        Row.fromSeq(row.toSeq :+ vector)
      }
    }

    // Update the schema for the new column
    val vectorizedColName = tempColumnName(columnName)
    val schema = StructType(df.schema.fields :+
      StructField(vectorizedColName, new VectorUDT()))

    sqlContext.createDataFrame(vectorized, schema)
      .drop(columnName)
      .withColumnRenamed(vectorizedColName, columnName)
  }

  /**
   * Converts the feature columns consisting of name, term, and value tuples into feature vectors
   *
   * @param df the source DataFrame
   * @param indexMapLoaders the index map loaders for all the columns to convert
   * @return the transformed DataFrame
   */
  protected def vectorize(df: DataFrame, indexMapLoaders: Map[String, IndexMapLoader]): DataFrame =
    indexMapLoaders.keys.foldLeft(df) { (acc, columnName) =>
      vectorize(acc, columnName, indexMapLoaders(columnName))
    }

  /**
   * Generates a default index map loader by scanning the dataframe
   *
   * @param df the source DataFrame
   * @param columnName the feature column from which to build the index
   * @return the index map loader
   */
  protected def generateIndexMapLoader(df: DataFrame, columnName: String): IndexMapLoader = {
    val features = df.select(columnName).flatMap { row: Row =>
      row.getSeq[Row](0).map {
        case Row(featureKey: String, _) => featureKey
      }
    }

    val featuresWithIntercept = features.distinct.collect :+ INTERCEPT_KEY

    val loader = new DefaultIndexMapLoader(featuresWithIntercept.zipWithIndex.toMap)
    loader.prepare(sc, params = null, namespace = IndexMap.GLOBAL_NS)

    loader
  }

  /**
   * Generates default index map loaders by scanning the dataframe
   *
   * @param df the source DataFrame
   * @param columns the feature columns from which to build the indices
   * @return the index map loader for each column
   */
  protected def generateIndexMapLoaders(df: DataFrame, columns: Set[String]): Map[String, IndexMapLoader] =
    columns.map { columnName => (columnName, generateIndexMapLoader(df, columnName)) }.toMap
}

object AvroDataReader {
  /**
   * Feature field names
   */
  object FieldNames {
    val NAME = "name"
    val TERM = "term"
    val VALUE = "value"
  }

  /**
   * Name, term, and feature key of the intercept
   */
  val INTERCEPT_NAME = "(INTERCEPT)"
  val INTERCEPT_TERM = ""
  val INTERCEPT_KEY = Utils.getFeatureKey(INTERCEPT_NAME, INTERCEPT_TERM)

  /**
   * Generates a temporary column name
   *
   * @param name the base name for the column
   * @return the temporary column name
   */
  protected def tempColumnName(name: String): String = s"__${name}__${System.currentTimeMillis}__"

  /**
   * Read a feature tuple from the name, term, and value DataFrame row
   *
   * @param feature the row from which to read the name, term, value tuple
   * @return the feature tuple
   */
  protected def readFeature(feature: Row): Option[Tuple2[String, Double]] = {
    val fieldNames = feature.schema.fieldNames

    if (fieldNames.contains(FieldNames.NAME) && fieldNames.contains(FieldNames.VALUE)) {
      val name = feature.getAs[String](FieldNames.NAME)

      val term = if (fieldNames.contains(FieldNames.TERM)) {
        feature.getAs[String](FieldNames.TERM)
      } else {
        ""
      }

      val value = feature.getAs[Number](FieldNames.VALUE).doubleValue

      Some(Utils.getFeatureKey(name, term) -> value)
    } else {
      None
    }
  }

  /**
   * A Spark sql udf that merges the features of all the columns into a single column
   *
   * @param cols Spark sql columns to be merged
   */
  protected def mergeFeatures = udf((cols: Seq[Seq[Row]]) =>
    cols.flatMap { features => features.flatMap(AvroDataReader.readFeature(_)) })

  /**
   * Drops all the specified columns from the DataFrame
   *
   * @param df the source DataFrame
   * @param cols the column names to drop
   * @return the transformed DataFrame
   */
  protected def dropColumns(df: DataFrame, cols: Seq[String]): DataFrame =
    cols.foldLeft(df) { case (acc, columnName) => acc.drop(columnName) }

  /**
   * Merges the feature columns according to the feature column map
   *
   * @param df the source DataFrame
   * @param featureColumnMap a map that specifies how the feature columns should be merged. The keys specify the name
   *   of the merged destination column, and the values are sets of source columns to merge, e.g.:
   *
   *     Map("userFeatures" -> Set("profileFeatures", "titleFeatures"))
   *
   *   This configuration merges the "profileFeatures" and "titleFeatures" columns into a single column named
   *   "userFeatures".
   * @return the transformed DataFrame
   */
  protected def mergeColumns(
      df: DataFrame,
      featureColumnMap: Map[String, Set[String]]): DataFrame = {

    val initTempColMappings = Map.empty[String, String]

    // Merge the columns into new temporary columns. We need to create intermediate temporary columns to correctly
    // handle cases where destination columns are named the same as source columns. Replacing these in-situ could cause
    // unexpected results when merging subsequent columns.
    val (merged, tempColMappings) = featureColumnMap.foldLeft((df, initTempColMappings)) {
      case ((currMerged, currTempColMappings), (destColumn, sourceColumns)) =>
        // This only specifies the columns. The columns are actually queried and results materialized later when we
        // evaluate the fully merged DataFrame
        val sourceCols = sourceColumns.map(col(_)).toSeq
        val newCol = mergeFeatures(array(sourceCols:_*))
        val tempColName = tempColumnName(destColumn)

        (currMerged.withColumn(tempColName, newCol),
          currTempColMappings ++ Map(destColumn -> tempColName))
    }

    // Remove the source feature columns
    val scrubbed = dropColumns(merged, featureColumnMap.values.flatten.toSeq.distinct)

    // Rename the temporary columns
    tempColMappings.foldLeft(scrubbed) { case (data, (destColumn, tempColName)) =>
      data.drop(destColumn)
        .withColumnRenamed(tempColName, destColumn)
    }
  }
}
