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

import java.util.{List => JList}

import scala.collection.JavaConverters._

import org.apache.avro.Schema
import org.apache.avro.Schema.Type._
import org.apache.avro.generic.GenericRecord
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.{Vector, VectorUDT}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.DataTypes._
import org.apache.spark.sql.types.{MapType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SQLContext}

import com.linkedin.photon.ml.avro.{AvroFieldNames, AvroUtils}
import com.linkedin.photon.ml.util.{DefaultIndexMapLoader, IndexMap, IndexMapLoader, Utils, VectorUtils}

/**
 * A DataReader implementation that reads the Avro format
 *
 * Note: This implementation replaces a previous implementation that used spark-avro for reading the Avro format. The
 * reasons for directly consuming avro here are twofold:
 *
 *    1) Spark-avro used a lot more memory on the production grid (cause for this still not identified)
 *    2) Using spark-avro caused a lot of shuffling, which made data reading much slower. The shuffling happened because
 *       the usual pattern for reading a DataFrame hides partition counts from the user, treating this as an
 *       implementation detail that a consumer shouldn't need to worry about at the DataFrame level of abstraction. This
 *       is usually true, but in the case of Photon, we still use partition counts in a few places to tune the
 *       performance of algorithms. The DataFrame runtime has no knowledge of this and badly underestimates the number
 *       of partitions needed, which causes a huge repartition/shuffle step during reading. Here we pass the partition
 *       count explicitly to the "read" function to get around this problem. Once we revisit Photon's sensitivity to
 *       choice of partition count, we can remove this parameter.
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
   * @param paths the paths to the avro files or folders
   * @param featureColumnMap a map that specifies how the feature columns should be merged. The keys specify the name
   *   of the merged destination column, and the values are sets of source columns to merge, e.g.:
   *
   *     Map("userFeatures" -> Set("profileFeatures", "titleFeatures"))
   *
   *   This configuration merges the "profileFeatures" and "titleFeatures" columns into a single column named
   *   "userFeatures".
   * @param numPartitions the minimum number of partitions. Spark is generally moving away from manually specifying
   *   partition counts like this, in favor of inferring it. However, Photon currently still exposes partition counts as
   *   a means for tuning job performance. The auto-inferred counts are usually much lower than the necessary counts for
   *   Photon (especially GAME), so this caused a lot of shuffling when repartitioning from the auto-partitioned data
   *   to the GAME data. We expose this setting here to avoid the shuffling.
   * @return the loaded and transformed DataFrame
   */
  override def readMerged(
      paths: Seq[String],
      featureColumnMap: Map[String, Set[String]],
      numPartitions: Int): (DataFrame, Map[String, IndexMapLoader]) = {

    require(paths.nonEmpty, "No paths specified. You must specify at least one input path.")
    require(numPartitions > 0, "Partition count must be greater than zero.")

    val records = AvroUtils.readAvroFiles(sc, paths, numPartitions)
    val indexMapLoaders = generateIndexMapLoaders(records, featureColumnMap)

    (readMerged(records, indexMapLoaders, featureColumnMap, numPartitions), indexMapLoaders)
  }

  /**
   * Reads the avro file at the given path into a DataFrame, using the given index map for feature names. Merges source
   * columns into combined feature vectors as specified by the featureColumnMap argument. Often features are joined from
   * different sources, and it can be more scalable to combine them into problem-specific feature vectors that can be
   * independently distributed.
   *
   * @param paths the paths to the avro files or folders
   * @param indexMapLoaders a map of index map loaders, containing one loader for each merged feature column
   * @param featureColumnMap a map that specifies how the feature columns should be merged. The keys specify the name
   *   of the merged destination column, and the values are sets of source columns to merge, e.g.:
   *
   *     Map("userFeatures" -> Set("profileFeatures", "titleFeatures"))
   *
   *   This configuration merges the "profileFeatures" and "titleFeatures" columns into a single column named
   *   "userFeatures".
   * @param numPartitions the minimum number of partitions. Spark is generally moving away from manually specifying
   *   partition counts like this, in favor of inferring it. However, Photon currently still exposes partition counts as
   *   a means for tuning job performance. The auto-inferred counts are usually much lower than the necessary counts for
   *   Photon (especially GAME), so this caused a lot of shuffling when repartitioning from the auto-partitioned data
   *   to the GAME data. We expose this setting here to avoid the shuffling.
   * @return the loaded and transformed DataFrame
   */
  override def readMerged(
      paths: Seq[String],
      indexMapLoaders: Map[String, IndexMapLoader],
      featureColumnMap: Map[String, Set[String]],
      numPartitions: Int): DataFrame = {

    require(paths.nonEmpty, "No paths specified. You must specify at least one input path.")
    require(numPartitions > 0, "Partition count must be greater than zero.")

    val records = AvroUtils.readAvroFiles(sc, paths, numPartitions)

    readMerged(records, indexMapLoaders, featureColumnMap, numPartitions)
  }

  /**
   * Reads the avro records into a DataFrame, using the given index map for feature names. Merges source columns into
   * combined feature vectors as specified by the featureColumnMap argument. Often features are joined from different
   * sources, and it can be more scalable to combine them into problem-specific feature vectors that can be
   * independently distributed.
   *
   * @param records the source avro records
   * @param indexMapLoaders a map of index map loaders, containing one loader for each merged feature column
   * @param featureColumnMap a map that specifies how the feature columns should be merged. The keys specify the name
   *   of the merged destination column, and the values are sets of source columns to merge, e.g.:
   *
   *     Map("userFeatures" -> Set("profileFeatures", "titleFeatures"))
   *
   *   This configuration merges the "profileFeatures" and "titleFeatures" columns into a single column named
   *   "userFeatures".
   * @param numPartitions the minimum number of partitions. Spark is generally moving away from manually specifying
   *   partition counts like this, in favor of inferring it. However, Photon currently still exposes partition counts as
   *   a means for tuning job performance. The auto-inferred counts are usually much lower than the necessary counts for
   *   Photon (especially GAME), so this caused a lot of shuffling when repartitioning from the auto-partitioned data
   *   to the GAME data. We expose this setting here to avoid the shuffling.
   * @return the loaded and transformed DataFrame
   */
  protected def readMerged(
      records: RDD[GenericRecord],
      indexMapLoaders: Map[String, IndexMapLoader],
      featureColumnMap: Map[String, Set[String]],
      numPartitions: Int): DataFrame = {

    // Infer sql schema from avro schema
    val srcFeatureCols = featureColumnMap.values.flatten.toSet
    val schemaFields = inferSchemaFields(records)
      .filterNot(f => srcFeatureCols.contains(f.name))

    val rows = records.mapPartitions { iter =>
      // Initialize index maps. This needs to be done here because indexMapForRDD is potentially an expensive call, and
      // should be done no more than once per partition.
      val featureShardIdToIndexMap = indexMapLoaders.map { case (shardId, loader) =>
        (shardId, loader.indexMapForRDD())
      }

      iter.map { record =>
        // Read non-feature columns
        val cols = readColumnValuesFromRecord(record, schemaFields)

        // Read feature columns
        val featureCols = featureColumnMap.map { case (destCol, sourceCols) =>
          val featureMap = featureShardIdToIndexMap(destCol)
          readFeatureVectorFromRecord(
            record,
            sourceCols,
            featureMap)
        }

        // Create the new row with all columns
        Row.fromSeq(cols ++ featureCols)
      }
    }

    // Add schema fields for the feature vector columns
    val featureFields = featureColumnMap.map { case (destCol, _) =>
      StructField(destCol, new VectorUDT())
    }

    val sqlSchema = new StructType((schemaFields ++ featureFields).toArray)
    sqlContext.createDataFrame(rows, sqlSchema)
  }

  /**
   * Generates default index map loaders by scanning the data
   *
   * @param records the avro records to scan for features
   * @param featureColumnMap a map that specifies how the feature columns should be merged. The keys specify the name
   *   of the merged destination column, and the values are sets of source columns to merge, e.g.:
   *
   *     Map("userFeatures" -> Set("profileFeatures", "titleFeatures"))
   *
   *   This configuration merges the "profileFeatures" and "titleFeatures" columns into a single column named
   *   "userFeatures".
   * @return the index map loaders
   */
  protected def generateIndexMapLoaders(
      records: RDD[GenericRecord],
      featureColumnMap: Map[String, Set[String]]): Map[String, IndexMapLoader] = {

    // Read a flattened collection of tuples of (shardId, features)
    val featuresByShard = records.flatMap { record =>
      featureColumnMap.map { case (shardId, sourceCols) =>
        (shardId, readFeaturesFromRecord(record, sourceCols).map(_._1))
      }
    }

    featuresByShard
      .reduceByKey { case (a, b) => (a ++ b).distinct }
      .collect
      .toMap
      .mapValues { features => DefaultIndexMapLoader(sc, (features :+ INTERCEPT_KEY).toSeq) }
      // have to map identity here because mapValues produces a non-serializable map
      // https://issues.scala-lang.org/browse/SI-7005
      .map(identity)
  }
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
   * Reads feature keys and values from the avro generic record.
   *
   * @param record the avro record
   * @param fieldNames the fields from which to read features
   * @return array of (feature key, value) tuples
   */
  def readFeaturesFromRecord(
      record: GenericRecord,
      fieldNames: Set[String]): Array[(String, Double)] = {

    require(Option(record).nonEmpty, "Can't read features from an empty record.")
    require(fieldNames.nonEmpty, "No feature fields specified.")

    fieldNames
      .toSeq
      .flatMap { fieldName => Option(record.get(fieldName)) match {
        case Some(recordList: JList[_]) => recordList.asScala.toSeq
        case other => throw new IllegalArgumentException(
          s"Expected feature list $fieldName to be a Java List, found instead: ${other.getClass.getName}.")
      }}
      .map {
        case record: GenericRecord =>
          val nameAndTerm = AvroUtils.readNameAndTermFromGenericRecord(record)
          val featureKey = Utils.getFeatureKey(nameAndTerm.name, nameAndTerm.term)
          (featureKey -> Utils.getDoubleAvro(record, AvroFieldNames.VALUE))

        case other => throw new IllegalArgumentException(s"$other in features list is not a GenericRecord")
      }.toArray
  }

  /**
   * Reads a feature vector from the avro record.
   *
   * @param record the avro record
   * @param fieldNames the source fields from which to read features for the vector
   * @param featureMap the feature map, which maps feature keys to vector indices
   * @return the feature vector
   */
  def readFeatureVectorFromRecord(
      record: GenericRecord,
      fieldNames: Set[String],
      featureMap: IndexMap): Vector = {

    require(Option(record).nonEmpty, "Can't read features from an empty record.")
    require(fieldNames.nonEmpty, "No feature fields specified.")

    // Retrieve the features
    val features = readFeaturesFromRecord(record, fieldNames)

    // Check for duplicate features
    val duplicateFeatures = features
      .groupBy(_._1)
      .filter(_._2.length > 1)
      .map { case (k, v) => (k, v.map(_._2).toList) }
    require(duplicateFeatures.isEmpty, s"Duplicate features found: ${duplicateFeatures.toString}")

    // Resolve feature keys to indices
    val featuresWithIndices = features.flatMap { case (featureKey, value) =>
      if (featureMap.contains(featureKey)) {
        Some(featureMap.getIndex(featureKey) -> value)
      } else {
        None
      }
    }

    // Add intercept if necessary
    val addIntercept = featureMap.contains(INTERCEPT_KEY)
    val featuresWithIntercept = if (addIntercept) {
      featuresWithIndices ++ Array(featureMap.getIndex(INTERCEPT_KEY) -> 1.0)
    } else {
      featuresWithIndices
    }

    // Create feature vector
    VectorUtils.breezeToMllib(
      VectorUtils.convertIndexAndValuePairArrayToSparseVector(
        featuresWithIntercept, featureMap.featureDimension))
  }

  /**
   * Infers spark sql field schema by sampling a single record from the RDD of avro GenericRecord.
   *
   * @param records the avro records
   * @return the spark sql field schema
   */
  def inferSchemaFields(records: RDD[GenericRecord]): Seq[StructField] = records
    .map(r => r.getSchema
      .getFields
      .asScala
      .flatMap { f => avroTypeToSql(f.name, f.schema) })
    .take(1)
    .head

  /**
   * Converts the named avro field schema to an equivalent spark sql schema type.
   *
   * @param name the field name
   * @param avroSchema the avro schema for the field
   * @return spark sql schema for the field
   */
  def avroTypeToSql(name: String, avroSchema: Schema): Option[StructField] = avroSchema.getType match {
    case INT => Some(StructField(name, IntegerType, nullable = false))
    case STRING => Some(StructField(name, StringType, nullable = false))
    case BOOLEAN => Some(StructField(name, BooleanType, nullable = false))
    case DOUBLE => Some(StructField(name, DoubleType, nullable = false))
    case FLOAT => Some(StructField(name, FloatType, nullable = false))
    case LONG => Some(StructField(name, LongType, nullable = false))
    case MAP =>
      avroTypeToSql(name, avroSchema.getValueType).map { valueSchema =>
        StructField(
          name,
          MapType(StringType, valueSchema.dataType, valueContainsNull = valueSchema.nullable),
          nullable = false)
      }

    case UNION =>
      if (avroSchema.getTypes.asScala.exists(_.getType == NULL)) {
        // In case of a union with null, take the first non-null type for the value type
        val remainingUnionTypes = avroSchema.getTypes.asScala.filterNot(_.getType == NULL)
        if (remainingUnionTypes.size == 1) {
          avroTypeToSql(name, remainingUnionTypes.head).map(_.copy(nullable = true))
        } else {
          avroTypeToSql(name, Schema.createUnion(remainingUnionTypes.asJava)).map(_.copy(nullable = true))
        }

      } else avroSchema.getTypes.asScala.map(_.getType) match {
        case Seq(t1) =>
          avroTypeToSql(name, avroSchema.getTypes.get(0))
        case Seq(t1, t2) if Set(t1, t2) == Set(INT, LONG) =>
          Some(StructField(name, LongType, nullable = false))
        case Seq(t1, t2) if Set(t1, t2) == Set(FLOAT, DOUBLE) =>
          Some(StructField(name, DoubleType, nullable = false))
        case _ =>
          // Unsupported union type. Drop this for now.
          None
      }

    case _ =>
      // Unsupported avro field type. Drop this for now.
      None
  }

  /**
   * Read the fields from the avro record into column values according to the supplied spark sql schema.
   *
   * @param record the avro GenericRecord
   * @param schemaFields the spark sql schema to apply when reading the record
   * @return column values
   */
  def readColumnValuesFromRecord(record: GenericRecord, schemaFields: Seq[StructField]) = schemaFields
    .flatMap { field: StructField => field.dataType match {
      case IntegerType => Some(Utils.getIntAvro(record, field.name))
      case StringType => Some(Utils.getStringAvro(record, field.name, field.nullable))
      case BooleanType => Some(Utils.getBooleanAvro(record, field.name))
      case DoubleType => Some(Utils.getDoubleAvro(record, field.name))
      case FloatType => Some(Utils.getFloatAvro(record, field.name))
      case LongType => Some(Utils.getLongAvro(record, field.name))
      case MapType(_, _, _) => Some(Utils.getMapAvro(record, field.name, field.nullable).asScala)
      case _ =>
        // Unsupported field type. Drop this for now.
        None
    }
  }
}
