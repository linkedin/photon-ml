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
 * @note Tried to mix-in trait Map[A, +B], but that turned into a quagmire that is not worth it.
 */
class InputColumnsNames extends Serializable {

  // We store default or user-defined input columns names in this efficient Array
  private val array: Array[String] = InputColumnsNames.all.map(colName => colName.toString)

  /**
   * Same as Map.updated, but the types are fixed.
   *
   * @param key The key to update
   * @param value The value corresponding to key
   * @return A new InputColumnsNames instance containing the update
   */
  def updated(key: InputColumnsNames.Value, value: String): InputColumnsNames = { array(key.id) = value; this }

  /**
   * Same as Array.apply(i: Int), i.e. an accessor for elements.
   *
   * @param key The key to retrieve
   * @return The column name for that key
   */
  def apply(key: InputColumnsNames.Value): String = array(key.id)

  /**
   * Return a human-readable representation of an InputColumnsNames instance.
   *
   * @return A human-readable string
   */
  override def toString: String = InputColumnsNames.all.map(icn => s"${icn.toString}: ${array(icn.id)}").mkString(", ")
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
   * @return An instance of class InputColumnsNames with default column names
   */
  def apply(): InputColumnsNames = new InputColumnsNames
}
