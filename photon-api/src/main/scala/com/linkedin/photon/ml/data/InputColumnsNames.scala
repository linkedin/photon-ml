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

/**
 * A class to hold user-defined input columns names.
 *
 * The default values are defined in the companion object. Any of these defaults can be overridden by the user (see
 * GameParams).
 *
 * @param array Array of input column names
 */
class InputColumnsNames private (array: Array[String]) extends Serializable {

  // Check that there are no duplicate column names
  require(array.length == getNames.size, "Each column must have unique a name.")

  /**
   * Same as Array.apply(i: Int), i.e. an accessor for elements.
   *
   * @param key The key to retrieve
   * @return The column name for that key
   */
  def apply(key: InputColumnsNames.Value): String = array(key.id)

  /**
   * Get the set of column names reserved for required columns.
   *
   * @return The input column names as a [[Set]]
   */
  def getNames: Set[String] = array.toSet

  /**
   * Returns a hash code value for the object.
   *
   * @return An [[Int]] hash code
   */
  override def hashCode: Int = array.hashCode()

  /**
   * Compare equality between this and some other object.
   *
   * @param obj Some other object
   * @return True if the other object is a [[InputColumnsNames]] with identical column names, otherwise false
   */
  override def equals(obj: Any): Boolean = obj match {
    case that: InputColumnsNames =>
      InputColumnsNames.all.forall { column =>
        this(column).equals(that(column))
      }

    case _ =>
      false
  }
}

/**
 * Internal names of the input columns used when reading datasets.
 * They can be changed by the user in the parameters.
 *
 * @note RESPONSE is required, together with features in input data, all other columns are optional
 * @note Although there is a default feature column name here, there can be multiple feature bags, each with their own
 *       name.
 */
object InputColumnsNames extends Enumeration {

  val UID = Value("uid")
  val RESPONSE = Value("response")
  val OFFSET = Value("offset")
  val WEIGHT = Value("weight")
  val META_DATA_MAP = Value("metadataMap")
  val FEATURES_DEFAULT = Value("features")

  // FEATURES_DEFAULT is excluded because feature shard names rely on another mechanism (see AvroDataReader)
  val all = Array(UID, RESPONSE, OFFSET, WEIGHT, META_DATA_MAP)

  /**
   * Constructor for instances of class InputColumnsNames.
   *
   * By default, they contain the names defined just above for the input column names.
   *
   * @param customNames [[Map]] of column to custom name
   * @return An instance of class InputColumnsNames with the given column names set, and default column names otherwise
   */
  def apply(customNames: Map[Value, String] = Map()): InputColumnsNames = {

    val array = all.map(_.toString)
    customNames.foreach { case (column, name) =>
      array(column.id) = name
    }

    new InputColumnsNames(array)
  }
}
