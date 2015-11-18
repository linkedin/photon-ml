/*
 * Copyright 2015 LinkedIn Corp. All rights reserved.
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

import breeze.linalg.{DenseVector, SparseVector, Vector}
import com.linkedin.photon.ml.io
import com.linkedin.photon.ml.io.GLMSuite
import org.apache.avro.generic.GenericRecord
import org.apache.avro.util.Utf8
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path

/**
 * Some useful functions
 * @author xazhang
 * @author nkatariy
 */
protected[ml] object Utils {

  /**
   * Get the feature key from an Avro generic record as key = name + delimiter + term
   * @param record The Avro generic record
   * @param delimiter The delimiter used to combine/separate name and term.
   * @return The feature name
   */
  def getFeatureKey(record: GenericRecord, nameKey: String, termKey: String, delimiter: String): String = {
    val name = getStringAvro(record, nameKey, isNullOK = false)
    val term = getStringAvro(record, termKey, isNullOK = true)
    getFeatureKey(name, term, delimiter)
  }

  /**
   * Get the feature key as a concatenation of name and term delimited by
   * [[io.GLMSuite.DELIMITER]]
   *
   * @param name feature name
   * @param term feature term
   * @return feature key
   */
  def getFeatureKey(name: String, term: String, delimiter: String = GLMSuite.DELIMITER): String = {
    name + delimiter + term
  }

  /**
   * Get the feature name from the feature key, expected to be formed using one of the [[getFeatureKey()]] methods
   *
   * @param key feature key
   * @param delimiter delimiter used to form the key. Default value is [[GLMSuite.DELIMITER]]
   * @return the feature name
   */
  def getFeatureNameFromKey(key: String, delimiter: String = GLMSuite.DELIMITER): String = {
    val tokens = key.split(delimiter)
    if (tokens.size != 2) {
      throw new IllegalArgumentException(s"Provided input [$key] is not a valid feature key")
    }
    tokens(0)
  }

  /**
   * Get the feature term from the feature key, expected to be formed using one of the [[getFeatureKey()]] methods
   *
   * @param key feature key
   * @param delimiter delimiter used to form the key. Default value is [[GLMSuite.DELIMITER]]
   * @return the feature term
   */
  def getFeatureTermFromKey(key: String, delimiter: String = GLMSuite.DELIMITER): String = {
    val tokens = key.split(delimiter)
    if (tokens.size != 2) {
      throw new IllegalArgumentException(s"Provided input [$key] is not a valid feature key")
    }
    tokens(1)
  }

  /**
   * Extract the String typed field with a given key from the Avro GenericRecord
   * @param record the generic record
   * @param key the key of the field
   * @return the String typed field
   */
  def getStringAvro(record: GenericRecord, key: String, isNullOK: Boolean = true): String = {
    record.get(key) match {
      case id@(_: Utf8 | _: JString) => id.toString
      case number: JNumber => number.toString
      case obj: JObject => throw new IllegalArgumentException(s"$key = $obj is neither a java String or Utf8")
      case _ => if (isNullOK) "" else throw new IllegalArgumentException(s"$key is null")
    }
  }

  /**
   * Extract the Double typed field with a given key from the Avro GenericRecord
   * @param record the generic record
   * @param key the key of the field
   * @return the Double typed field
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
   * Parse String to Double
   */
  private def atod(string: String): Double = {
    if (string.length() < 1)
      throw new IllegalArgumentException("Can't convert empty string to double")
    val double = string.toDouble
    if (JDouble.isNaN(double) || JDouble.isInfinite(double)) {
      throw new IllegalArgumentException(s"NaN or Infinity in input: $string")
    }
    double
  }

  /**
   * Extract the Float typed field with a given key from the Avro GenericRecord
   * @param record the generic record
   * @param key the key of the field
   * @return the Float typed field
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
   * Parse String to Float
   */
  private def atof(string: String): Float = {
    if (string.length() < 1)
      throw new IllegalArgumentException("Can't convert empty string to float")
    val float = string.toFloat
    if (JFloat.isNaN(float) || JFloat.isInfinite(float)) {
      throw new IllegalArgumentException(s"NaN or Infinity in input: $string")
    }
    float
  }

  /**
   * Extract the Int typed field with a given key from the Avro GenericRecord
   * @param record the generic record
   * @param key the key of the field
   * @return the Int typed field
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
   * Extract the Long typed field with a given key from the Avro GenericRecord
   * @param record the generic record
   * @param key the key of the field
   * @return the Long typed field
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
   * Extract the Boolean typed field with a given key from the Avro GenericRecord
   * @param record the generic record
   * @param key the key of the field
   * @return the Boolean typed field
   */
  def getBooleanAvro(record: GenericRecord, key: String): Boolean = {
    record.get(key) match {
      case booleanValue: JBoolean => booleanValue.booleanValue
      // Note: Scala String#toBoolean method is better than JBoolean#parseBoolean in the sense that it only accepts
      // "true" or "false" (case-insensitive) and throw exceptions for other string values.
      case id@(_: Utf8 | _: JString) => id.toString.toBoolean
      case obj: JObject => throw new IllegalArgumentException(s"$key = $obj is an unknown object")
      case _ => throw new IllegalArgumentException(s"$key is null")
    }
  }

  /**
   * Delete a given directory on HDFS, it will be silent if the directory does not exist
   *
   * @param dir the directory path
   * @param hadoopConf the Hadoop Configuration object
   */
  def deleteHDFSDir(dir: String, hadoopConf: Configuration): Unit = {
    val path = new Path(dir)
    val fs = path.getFileSystem(hadoopConf)
    if (fs.exists(path)) fs.delete(path, true)
  }

  /**
   * Create a HDFS directory, it will be silent if the directory already exists
   *
   * @param dir the directory path
   * @param hadoopConf the Hadoop Configuration object
   */
  def createHDFSDir(dir: String, hadoopConf: Configuration): Unit = {
    val path = new Path(dir)
    val fs = path.getFileSystem(hadoopConf)
    if (!fs.exists(path)) fs.mkdirs(path)
  }

  /**
   * This function is copied from MLlib's MLUtils.log1pExp
   * When `x` is positive and large, computing `math.log(1 + math.exp(x))` will lead to arithmetic
   * overflow. This will happen when `x > 709.78` which is not a very large number.
   * It can be addressed by rewriting the formula into `x + math.log1p(math.exp(-x))` when `x > 0`.
   *
   * @param x a floating-point value as input.
   * @return the result of `math.log(1 + math.exp(x))`.
   */
  def log1pExp(x: Double): Double = {
    if (x > 0) {
      x + math.log1p(math.exp(-x))
    } else {
      math.log1p(math.exp(x))
    }
  }

  /**
   * Initialize the a zeros vector of the same type as the input prototype vector. I.e., if the prototype vector is
   * a sparse vector, then the initialized zeros vector should also be initialized as a sparse vector, and if the
   * prototype vector is a dense vector, then the initialized zeros vector should also be initialized as a dense vector.
   * @param prototypeVector The input prototype vector
   * @return The initialized vector
   */
  def initializeZerosVectorOfSameType(prototypeVector: Vector[Double]): Vector[Double] = {
    prototypeVector match {
      case dense: DenseVector[Double] => DenseVector.zeros[Double](dense.length)
      case sparse: SparseVector[Double] => new SparseVector[Double](sparse.array.index, Array.fill(sparse.array.index.length)(0.0), sparse.length)
      case _ => throw new IllegalArgumentException("Vector types other than " + classOf[DenseVector[Double]]
          + " and " + classOf[SparseVector[Double]] + " is not supported. Current class type: "
          + prototypeVector.getClass.getName + ".")
    }
  }

  /**
   * This is a slight modification of the default getOrElse method provided by scala.
   *
   * The method looks up the key in the given map from [[String]] to [[Any]]. If it finds something of the provided
   * generic type [[T]], returns it. Otherwise, depending on the contents of the input [[Either]], an exception is
   * thrown or a default value is returned.
   *
   * @param map input map to look up
   * @param key the key to be looked up in the provided map
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
      case Some(x: T) => x
      case _ =>
        elseBranch match {
          case Left(errorMsg) => throw new IllegalArgumentException(errorMsg)
          case Right(defaultValue) => defaultValue
        }
    }
  }
}