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
package com.linkedin.photon.ml.util

import java.lang.{Boolean => JBoolean, Double => JDouble, Float => JFloat, Number => JNumber, Object => JObject, String => JString}

import scala.collection.JavaConverters._

import org.apache.avro.generic.GenericRecord
import org.apache.avro.util.Utf8
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path

import com.linkedin.photon.ml.Constants
import com.linkedin.photon.ml.evaluation.EvaluatorType._
import com.linkedin.photon.ml.evaluation.{MultiAUC, MultiPrecisionAtK, EvaluatorType}

// TODO: Better documentation.

/**
 * Some useful functions
 */
object Utils {

  /**
   * Get the feature key from an Avro generic record as key = name + delimiter + term.
   *
   * @param record The Avro generic record
   * @param delimiter The delimiter used to combine/separate name and term.
   * @return The feature name
   */
  def getFeatureKey(record: GenericRecord, nameKey: String, termKey: String, delimiter: String): String = {
    val name = getStringAvro(record, nameKey)
    val term = getStringAvro(record, termKey, isNullOK = true)
    getFeatureKey(name, term, delimiter)
  }

  /**
   * Get the feature key as a concatenation of name and term delimited by
   * [[com.linkedin.photon.ml.Constants.DELIMITER]].
   *
   * @param name Feature name
   * @param term Feature term
   * @return Feature key
   */
  def getFeatureKey(name: CharSequence, term: CharSequence, delimiter: String = Constants.DELIMITER): String =
    name + delimiter + term

  /**
   * Get the feature name from the feature key, expected to be formed using one of the [[getFeatureKey()]] methods.
   *
   * @param key Feature key
   * @param delimiter Delimiter used to form the key. Default value is [[Constants.DELIMITER]]
   * @return The feature name
   */
  def getFeatureNameFromKey(key: String, delimiter: String = Constants.DELIMITER): String = {
    require(delimiter.r.findAllIn(key).length == 1, s"Provided input [$key] is not a valid feature key")
    key.split(delimiter).headOption.getOrElse("")
  }

  /**
   * Get the feature term from the feature key, expected to be formed using one of the [[getFeatureKey()]] methods.
   *
   * @param key Feature key
   * @param delimiter Delimiter used to form the key. Default value is [[Constants.DELIMITER]]
   * @return The feature term
   */
  def getFeatureTermFromKey(key: String, delimiter: String = Constants.DELIMITER): String = {
    require(delimiter.r.findAllIn(key).length == 1, s"Provided input [$key] is not a valid feature key")
    key.split(delimiter).lift(1).getOrElse("")
  }

  /**
   * Extract the String typed field with a given key from the Avro GenericRecord.
   *
   * @param record The generic record
   * @param key The key of the field
   * @param isNullOK Whether null is accepted. If set to true, then an empty string will be returned if the
   *                 corresponding field of the key is null, otherwise, exception will be thrown.
   * @return The String typed field
   */
  def getStringAvro(record: GenericRecord, key: String, isNullOK: Boolean = false): String = {
    record.get(key) match {
      case id@(_: Utf8 | _: JString) => id.toString
      case number: JNumber => number.toString
      case obj: JObject => throw new IllegalArgumentException(s"$key = $obj is neither a java String or Utf8")
      case _ => if (isNullOK) "" else throw new IllegalArgumentException(s"$key is null")
    }
  }

  /**
   * Extract the Double typed field with a given key from the Avro GenericRecord.
   *
   * @param record The generic record
   * @param key The key of the field
   * @return The Double typed field
   */
  def getDoubleAvro(record: GenericRecord, key: String): Double = {
    record.get(key) match {
      case number: JNumber => number.doubleValue
      case id@(_: Utf8 | _: JString) => atod(id.toString)
      case obj: JObject => throw new IllegalArgumentException(s"$key = $obj is an unknown object")
      case _ => throw new IllegalArgumentException(s"$key is null")
    }
  }

  /**
   * Fetch the java map from an Avro map field.
   *
   * @param record The Avro generic record
   * @param field The field key
   * @return A java map of String -> Object
   */
  def getMapAvro(
      record: GenericRecord,
      field: String,
      isNullOK: Boolean = false): Map[String, JObject] = {

    val map = record.get(field).asInstanceOf[java.util.Map[Any, JObject]]

    if (map == null && isNullOK) {
      null
    } else if (map == null) {
      throw new IllegalArgumentException(s"field '$field' is null")
    } else {
      map
        .asScala
        .flatMap { case (key, value) =>

          val keyString = key.toString

          value match {
            // Need to convert Utf8 values to String here, because otherwise we get schema casting errors and misleading
            // equivalence failures downstream.
            case s@(_: Utf8 | _: JString) => Some((keyString, s.toString))
            case x@(_: Number  | _: JBoolean) => Some((keyString, x))
            case _ => None
          }
        }
        .toMap
    }
  }

  /**
   * Parse String to Double.
   *
   * @param string The string to parse
   * @return The double parsed from the string, or an exception if string is empty or double is NaN or Infinity
   */
  private def atod(string: String): Double = {
    if (string.length() < 1) {
      throw new IllegalArgumentException("Can't convert empty string to double")
    }

    val double = string.toDouble
    if (JDouble.isNaN(double) || JDouble.isInfinite(double)) {
      throw new IllegalArgumentException(s"NaN or Infinity in input: $string")
    }

    double
  }

  /**
   * Extract the Float typed field with a given key from the Avro GenericRecord.
   *
   * @param record The generic record
   * @param key The key of the field
   * @return The Float typed field
   */
  def getFloatAvro(record: GenericRecord, key: String): Float = {
    record.get(key) match {
      case number: JNumber => number.floatValue
      case id@(_: Utf8 | _: JString) => atof(id.toString)
      case obj: JObject => throw new IllegalArgumentException(s"$key = $obj is an unknown object")
      case _ => throw new IllegalArgumentException(s"$key is null")
    }
  }

  /**
   * Parse String to Float.
   *
   * @param string The string to parse
   * @return A float parse from the string, or an exception if the string is empty or the flat is NaN or Infinity
   */
  private def atof(string: String): Float = {
    if (string.length() < 1) {
      throw new IllegalArgumentException("Can't convert empty string to float")
    }

    val float = string.toFloat
    if (JFloat.isNaN(float) || JFloat.isInfinite(float)) {
      throw new IllegalArgumentException(s"NaN or Infinity in input: $string")
    }

    float
  }

  /**
   * Extract the Int typed field with a given key from the Avro GenericRecord.
   *
   * @param record The generic record
   * @param key The key of the field
   * @return The Int typed field
   */
  def getIntAvro(record: GenericRecord, key: String): Int = {
    record.get(key) match {
      case number: JNumber => number.intValue
      case id@(_: Utf8 | _: JString) => id.toString.toInt
      case obj: JObject => throw new IllegalArgumentException(s"$key = $obj is an unknown object")
      case _ => throw new IllegalArgumentException(s"$key is null")
    }
  }

  /**
   * Extract the Long typed field with a given key from the Avro GenericRecord.
   *
   * @param record The generic record
   * @param key The key of the field
   * @return The Long typed field
   */
  def getLongAvro(record: GenericRecord, key: String): Long = {
    record.get(key) match {
      case number: JNumber => number.longValue()
      case id@(_: Utf8 | _: JString) => id.toString.toLong
      case obj: JObject => throw new IllegalArgumentException(s"$key = $obj is an unknown object")
      case _ => throw new IllegalArgumentException(s"$key is null")
    }
  }

  /**
   * Extract the Boolean typed field with a given key from the Avro GenericRecord.
   *
   * @param record The generic record
   * @param key The key of the field
   * @return The Boolean typed field
   */
  def getBooleanAvro(record: GenericRecord, key: String): Boolean = {
    record.get(key) match {
      case booleanValue: JBoolean => booleanValue.booleanValue
      // NOTE Scala String#toBoolean method is better than JBoolean#parseBoolean in the sense that it only accepts
      // "true" or "false" (case-insensitive) and throw exceptions for other string values.
      case id@(_: Utf8 | _: JString) => id.toString.toBoolean
      case obj: JObject => throw new IllegalArgumentException(s"$key = $obj is an unknown object")
      case _ => throw new IllegalArgumentException(s"$key is null")
    }
  }

  /**
   * Delete a given directory on HDFS, it will be silent if the directory does not exist.
   *
   * @param dir The directory path
   * @param hadoopConf The Hadoop Configuration object
   */
  def deleteHDFSDir(dir: Path, hadoopConf: Configuration): Unit = {
    val fs = dir.getFileSystem(hadoopConf)
    if (fs.exists(dir)) fs.delete(dir, true)
  }

  /**
   * Create a HDFS directory, it will be silent if the directory already exists.
   *
   * @param dir The directory path
   * @param hadoopConf The Hadoop Configuration object
   */
  def createHDFSDir(dir: Path, hadoopConf: Configuration): Unit = {
    val fs = dir.getFileSystem(hadoopConf)
    if (!fs.exists(dir)) fs.mkdirs(dir)
  }

  /**
   * This is a slight modification of the default getOrElse method provided by scala.
   *
   * The method looks up the key in the given map from [[String]] to [[Any]]. If it finds something of the provided
   * generic type [[T]], returns it. Otherwise, depending on the contents of the input [[Either]], an exception is
   * thrown or a default value is returned.
   *
   * @param map Input map to look up
   * @param key The key to be looked up in the provided map
   * @param elseBranch If one wants to fail on not finding a value of type [[T]] in the map, an
   *                   [[IllegalArgumentException]] will be thrown with the error message provided. If one wants to
   *                   continue without failure, a default value is expected that will be returned
   * @tparam T Intended return type of the method
   * @throws java.lang.IllegalArgumentException Exception thrown if a value of type [[T]] isn't found in the map and
   *                                            the error message is non-empty
   * @return A value of type [[T]] or throw an [[IllegalArgumentException]]
   */
  @throws(classOf[IllegalArgumentException])
  def getKeyFromMapOrElse[T](map: Map[String, Any], key: String, elseBranch: Either[String, T]): T = {
    map.get(key) match {
      case Some(x) => x.asInstanceOf[T]
      case _ =>
        elseBranch match {
          case Left(errorMsg) => throw new IllegalArgumentException(errorMsg)
          case Right(defaultValue) => defaultValue
        }
    }
  }

  /**
   * Parse the evaluator type with name.
   *
   * @param name Name of the evaluator type
   * @return The parsed evaluator type
   */
  def evaluatorWithName(name: String): EvaluatorType = name.trim.toUpperCase match {
    case AUC.name => AUC
    case AUPR.name => AUPR
    case RMSE.name => RMSE
    case LogisticLoss.name | "LOGISTICLOSS" => LogisticLoss
    case PoissonLoss.name | "POISSONLOSS" => PoissonLoss
    case SmoothedHingeLoss.name | "SMOOTHEDHINGELOSS" => SmoothedHingeLoss
    case SquaredLoss.name | "SQUAREDLOSS" => SquaredLoss
    case MultiPrecisionAtK.batchPrecisionAtKPattern(k, _) =>
      val MultiPrecisionAtK.batchPrecisionAtKPattern(_, idName) = name.trim
      MultiPrecisionAtK(k.toInt, idName)
    case MultiAUC.batchAUCPattern(_) =>
      val MultiAUC.batchAUCPattern(idName) = name.trim
      MultiAUC(idName)
    case _ => throw new IllegalArgumentException(s"Unsupported evaluator $name")
  }

  /**
   * This avoids if statements in the code.
   *
   * @param p A predicate: if it is true, call f (and wrap result in Option)
   * @param f A function we want to call, only if p
   * @tparam T The return type of f
   * @return Some[T] if p or None
   */
  def filter[T](p: => Boolean)(f: => T): Option[T] = if (p) Some(f) else None
}
