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
package com.linkedin.photon.ml.data.avro

import java.util.{List => JList}

import scala.collection.JavaConverters._

import org.apache.avro.Schema
import org.apache.avro.Schema.Type._
import org.apache.avro.generic.GenericRecord
import org.apache.spark.SparkContext
import org.apache.spark.ml.linalg.{SparseVector, Vector}
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.DataTypes._
import org.apache.spark.sql.types.{MapType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SQLContext}

import com.linkedin.photon.ml.Constants
import com.linkedin.photon.ml.data.{DataReader, InputColumnsNames}
import com.linkedin.photon.ml.index.{DefaultIndexMapLoader, IndexMap, IndexMapLoader}
import com.linkedin.photon.ml.io.FeatureShardConfiguration
import com.linkedin.photon.ml.util._

/**
 * A DataReader implementation that reads the Avro format.
 *
 * @note This implementation replaces a previous implementation that used spark-avro for reading the Avro format. The
 *       reasons for directly consuming avro here are twofold:
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
 * @param sc The Spark context
 * @param defaultFeatureColumn The default column to use for features
 */
class AvroDataReader(
    sc: SparkContext,
    defaultFeatureColumn: String = InputColumnsNames.FEATURES_DEFAULT.toString)
  extends DataReader(defaultFeatureColumn) {

  import AvroDataReader._

  /**
   * Internal sql context
   */
  private val sqlContext = new SQLContext(sc)

  /**
   * Reads the avro files at the given paths into a DataFrame, generating a default index map for feature names. Merges
   * source columns into combined feature vectors as specified by the featureColumnMap argument. Often features are
   * joined from different sources, and it can be more scalable to combine them into problem-specific feature vectors
   * that can be independently distributed.
   *
   * @param paths The path to the files or folders
   * @param featureColumnConfigsMap A map that specifies how the feature columns should be merged. The keys specify the
   *                                name of the merged destination column, and the values are configs containing sets of
   *                                source columns to merge, e.g.:
   *
   *   Map("userFeatures" -> FeatureShardConfiguration(Set("profileFeatures", "titleFeatures")))
   *
   *                                This configuration merges the "profileFeatures" and "titleFeatures" columns into a
   *                                single column named "userFeatures".
   * @param numPartitions The minimum number of partitions. Spark is generally moving away from manually specifying
   *                      partition counts like this, in favor of inferring it. However, Photon currently still exposes
   *                      partition counts as a means for tuning job performance. The auto-inferred counts are usually
   *                      much lower than the necessary counts for Photon (especially GAME), so this caused a lot of
   *                      shuffling when repartitioning from the auto-partitioned data to the GAME data. We expose this
   *                      setting here to avoid the shuffling.
   * @return The loaded and transformed DataFrame
   */
  override def readMerged(
      paths: Seq[String],
      featureColumnConfigsMap: Map[MergedColumnName, FeatureShardConfiguration],
      numPartitions: Int): (DataFrame, Map[MergedColumnName, IndexMapLoader]) = {

    require(paths.nonEmpty, "No paths specified. You must specify at least one input path.")
    require(numPartitions >= 0, "Partition count cannot be negative.")

    val records = AvroUtils.readAvroFiles(sc, paths, numPartitions)
    val featureColumnMap = featureColumnConfigsMap.mapValues(_.featureBags).map(identity)
    val interceptColumnMap = featureColumnConfigsMap.mapValues(_.hasIntercept).map(identity)
    val indexMapLoaders = generateIndexMapLoaders(records, featureColumnMap, interceptColumnMap)

    (readMerged(records, indexMapLoaders, featureColumnMap), indexMapLoaders)
  }

  /**
   * Reads the files at the given paths into a DataFrame, using the given index map for feature names. Merges source
   * columns into combined feature vectors as specified by the featureColumnMap argument. Often features are joined from
   * different sources, and it can be more scalable to combine them into problem-specific feature vectors that can be
   * independently distributed.
   *
   * @param paths The path to the files or folders
   * @param indexMapLoaders A map of index map loaders, containing one loader for each merged feature column
   * @param featureColumnConfigsMap A map that specifies how the feature columns should be merged. The keys specify the
   *                                name of the merged destination column, and the values are configs containing sets of
   *                                source columns to merge, e.g.:
   *
   *   Map("userFeatures" -> FeatureShardConfiguration(Set("profileFeatures", "titleFeatures")))
   *
   *                                This configuration merges the "profileFeatures" and "titleFeatures" columns into a
   *                                single column named "userFeatures".
   * @param numPartitions The minimum number of partitions. Spark is generally moving away from manually specifying
   *                      partition counts like this, in favor of inferring it. However, Photon currently still exposes
   *                      partition counts as a means for tuning job performance. The auto-inferred counts are usually
   *                      much lower than the necessary counts for Photon (especially GAME), so this caused a lot of
   *                      shuffling when repartitioning from the auto-partitioned data to the GAME data. We expose this
   *                      setting here to avoid the shuffling.
   * @return The loaded and transformed DataFrame
   */
  override def readMerged(
      paths: Seq[String],
      indexMapLoaders: Map[MergedColumnName, IndexMapLoader],
      featureColumnConfigsMap: Map[MergedColumnName, FeatureShardConfiguration],
      numPartitions: Int): DataFrame = {

    require(paths.nonEmpty, "No paths specified. You must specify at least one input path.")
    require(numPartitions >= 0, "Partition count cannot be negative.")

    val records = AvroUtils.readAvroFiles(sc, paths, numPartitions)
    val featureColumnMap = featureColumnConfigsMap.mapValues(_.featureBags).map(identity)

    readMerged(records, indexMapLoaders, featureColumnMap)
  }

  /**
   * Reads the avro records into a DataFrame, using the given index map for feature names. Merges source columns into
   * combined feature vectors as specified by the featureColumnMap argument. Often features are joined from different
   * sources, and it can be more scalable to combine them into problem-specific feature vectors that can be
   * independently distributed.
   *
   * @param records The source avro records
   * @param indexMapLoaders A map of index map loaders, containing one loader for each merged feature column
   * @param featureColumnMap A map that specifies how the feature columns should be merged. The keys specify the name
   *                         of the merged destination column, and the values are sets of source columns to merge, e.g.:
   *
   *     Map("userFeatures" -> Set("profileFeatures", "titleFeatures"))
   *
   *                         This configuration merges the "profileFeatures" and "titleFeatures" columns into a single
   *                         column named "userFeatures". "userFeatures" here is a "feature shard". "profileFeatures"
   *                         here is a "feature bag".
   * @return The loaded and transformed DataFrame
   */
  protected def readMerged(
      records: RDD[GenericRecord],
      indexMapLoaders: Map[MergedColumnName, IndexMapLoader],
      featureColumnMap: Map[MergedColumnName, Set[InputColumnName]]): DataFrame = {

    // Infer sql schema from avro schema
    val srcFeatureCols = featureColumnMap.values.flatten.toSet
    val schemaFields = inferSchemaFields(records)
      .getOrElse(throw new IllegalArgumentException("No records found at given input paths."))
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
          val featureMap: IndexMap = featureShardIdToIndexMap(destCol)

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
      StructField(destCol, VectorType)
    }
    val sqlSchema = new StructType((schemaFields ++ featureFields).toArray)

    sqlContext.createDataFrame(rows, sqlSchema)
  }

  /**
   * Generates default index map loaders by scanning the data.
   *
   * @param records The avro records to scan for features
   * @param interceptColumnMap A map of intercept settings, containing one setting for each merged feature column
   * @param featureColumnMap A map that specifies how the feature columns should be merged. The keys specify the names
   *   of the merged destination column, and the values are sets of source columns to merge, e.g.:
   *     Map("userFeatures" -> Set("profileFeatures", "titleFeatures"))
   *   This configuration merges the "profileFeatures" and "titleFeatures" columns into a single column named
   *   "userFeatures". "userFeatures" here is a "feature shard". "profileFeatures" here is a "feature bag".
   * @return The index map loaders
   */
  protected def generateIndexMapLoaders(
      records: RDD[GenericRecord],
      featureColumnMap: Map[MergedColumnName, Set[InputColumnName]],
      interceptColumnMap: Map[MergedColumnName, Boolean]): Map[MergedColumnName, IndexMapLoader] = {

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
      .map { case (shardId, features) =>

        val addIntercept = interceptColumnMap(shardId)
        val featureNames = if (addIntercept) {
          (features :+ Constants.INTERCEPT_KEY).toSeq
        } else {
          features.toSeq
        }

        (shardId, DefaultIndexMapLoader(sc, featureNames))
      }
  }
}

object AvroDataReader {

  // Avro to sql primitive type map
  private val primitiveTypeMap = Map(
    INT -> IntegerType,
    LONG -> LongType,
    FLOAT -> FloatType,
    DOUBLE -> DoubleType,
    STRING -> StringType,
    BOOLEAN -> BooleanType
  )

  /**
   * Establishes precedence among numeric types, for resolving unions where multiple types are specified. Appearing
   * earlier in the list means higher precedence.
   *
   * @note This doesn't need to be exactly right -- its purpose is to allow us to do something basically sensible when
   *       the data contains strange union types. Data specified with proper types are not affected.
   */
  private val numericPrecedence = List(DOUBLE, FLOAT, LONG, INT)

  /**
   * Reads feature keys and values from the avro generic record.
   *
   * @param record The avro record
   * @param fieldNames The fields from which to read features
   * @return Array of (feature key, value) tuples
   */
  protected[data] def readFeaturesFromRecord(
      record: GenericRecord,
      fieldNames: Set[String]): Array[(String, Double)] = {

    require(Option(record).nonEmpty, "Can't read features from an empty record.")

    fieldNames
      .toSeq
      .flatMap { fieldName =>
        Some(record.get(fieldName)) match {
          // Must have conversion to Seq at the end (labelled redundant by IDEA) or else typing compiler errors
          case Some(recordList: JList[_]) => recordList.asScala.toSeq
          case other => throw new IllegalArgumentException(
            s"Expected feature list $fieldName to be a Java List, found instead: ${other.getClass.getName}.")
        }
      }
      .map {
        case record: GenericRecord =>
          val nameAndTerm = AvroUtils.readNameAndTermFromGenericRecord(record)
          val featureKey = Utils.getFeatureKey(nameAndTerm.name, nameAndTerm.term)

          featureKey -> Utils.getDoubleAvro(record, TrainingExampleFieldNames.VALUE)

        case other => throw new IllegalArgumentException(s"$other in features list is not a GenericRecord")
      }.toArray
  }

  /**
   * Reads a feature vector from the avro record.
   *
   * @param record The avro record
   * @param fieldNames The source fields from which to read features for the vector
   * @param featureMap the feature map, which maps feature keys to vector indices
   * @return The feature vector
   */
  protected[data] def readFeatureVectorFromRecord(
      record: GenericRecord,
      fieldNames: Set[String],
      featureMap: IndexMap): Vector = {

    require(Option(record).nonEmpty, "Can't read features from an empty record.")

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
    val addIntercept = featureMap.contains(Constants.INTERCEPT_KEY)
    val featuresWithIntercept = if (addIntercept) {
      featuresWithIndices ++ Array(featureMap.getIndex(Constants.INTERCEPT_KEY) -> 1.0)
    } else {
      featuresWithIndices
    }

    // Create feature vector
    val (indices, values) = featuresWithIntercept.sorted.unzip
    new SparseVector(featureMap.featureDimension, indices.toArray, values.toArray)
  }

  /**
   * Infers spark sql field schema by sampling a single record from the RDD of avro GenericRecord.
   *
   * @param records The avro records
   * @return The spark sql field schema, or None if no records were found
   */
  protected[data] def inferSchemaFields(records: RDD[GenericRecord]): Option[Seq[StructField]] =
    records
      .map(r => r.getSchema
        .getFields
        .asScala
        .flatMap { f => avroTypeToSql(f.name, f.schema) })
      .take(1)
      .headOption

  /**
   * Determines whether all the types are numeric
   *
   * @param types the types to check
   * @return true if all specified types are numeric
   */
  protected[data] def allNumericTypes(types: Seq[Schema.Type]): Boolean =
    types.forall(numericPrecedence.toSet)

  /**
   * Selects the "dominant" avro numeric type from the list. Dominance in this sense means the numeric type with highest
   * precedence.
   *
   * @param types the avro types from which to select
   * @return the dominant numeric type
   */
  protected[data] def getDominantNumericType(types: Seq[Schema.Type]): Schema.Type =
    numericPrecedence.filter(types.toSet).head

  /**
   * Converts the named avro field schema to an equivalent spark sql schema type.
   *
   * @param name The field name
   * @param avroSchema The avro schema for the field
   * @return Spark sql schema for the field
   */
  protected[data] def avroTypeToSql(name: String, avroSchema: Schema): Option[StructField] =
    avroSchema.getType match {
      case avroType @ (INT | LONG | FLOAT | DOUBLE | STRING | BOOLEAN) =>
        Some(StructField(name, primitiveTypeMap(avroType), nullable = false))

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
          // When there are cases of multiple non-null types, resolve to a single sql type
          case Seq(_) =>
            avroTypeToSql(name, avroSchema.getTypes.get(0))

          case numericTypes if allNumericTypes(numericTypes) =>
            Some(StructField(name, primitiveTypeMap(getDominantNumericType(numericTypes)), nullable = false))

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
   * @param record The avro GenericRecord
   * @param schemaFields The spark sql schema to apply when reading the record
   * @return Column values
   */
  protected[data] def readColumnValuesFromRecord(record: GenericRecord, schemaFields: Seq[StructField]): Seq[Any] =

    schemaFields
      .flatMap { field: StructField =>
        field.dataType match {
          case IntegerType => checkNull(record, field).orElse(Some(Utils.getIntAvro(record, field.name)))
          case StringType => Some(Utils.getStringAvro(record, field.name, field.nullable))
          case BooleanType => checkNull(record, field).orElse(Some(Utils.getBooleanAvro(record, field.name)))
          case DoubleType => checkNull(record, field).orElse(Some(Utils.getDoubleAvro(record, field.name)))
          case FloatType => checkNull(record, field).orElse(Some(Utils.getFloatAvro(record, field.name)))
          case LongType => checkNull(record, field).orElse(Some(Utils.getLongAvro(record, field.name)))
          case MapType(_, _, _) => Some(Utils.getMapAvro(record, field.name, field.nullable))
          case _ =>
            // Unsupported field type. Drop this for now.
            None
        }
      }

  /**
   * Checks whether null values are allowed for the record, and if so, passes along the null value. Otherwise, returns
   * None.
   *
   * @param record The avro GenericRecord
   * @param field The schema field
   * @return Some(null) if the field is null and nullable. None otherwise.
   */
  protected[data] def checkNull(record: GenericRecord, field: StructField): Option[_] =

    if (record.get(field.name) == null && field.nullable) {
      Some(null)
    } else {
      None
    }
}
