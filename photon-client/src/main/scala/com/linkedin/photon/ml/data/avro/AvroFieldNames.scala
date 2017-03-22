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

/**
 * Common field names in the Avro data set used in Photon-ML.
 */
abstract class AvroFieldNames extends Serializable {

  val UID = AvroFieldNames.UID
  val NAME = AvroFieldNames.NAME
  val TERM = AvroFieldNames.TERM
  val FEATURES = AvroFieldNames.FEATURES
  val VALUE = AvroFieldNames.VALUE
  val RESPONSE: String
  val OFFSET = AvroFieldNames.OFFSET
  val WEIGHT = AvroFieldNames.WEIGHT
  val META_DATA_MAP = AvroFieldNames.META_DATA_MAP
}

object AvroFieldNames {

  val UID: String = "uid"
  val NAME: String = "name"
  val TERM: String = "term"
  val FEATURES: String = "features"
  val VALUE: String = "value"
  val RESPONSE: String = null
  val OFFSET: String = "offset"
  val WEIGHT: String = "weight"
  val META_DATA_MAP: String = "metadataMap"
}
