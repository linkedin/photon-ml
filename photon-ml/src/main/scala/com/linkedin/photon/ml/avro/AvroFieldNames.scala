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
package com.linkedin.photon.ml.avro

/**
 * Common field names in the Avro data set used in GAME
 */
object AvroFieldNames {
  protected[avro] val NAME: String = "name"
  protected[avro] val TERM: String = "term"
  protected[avro] val VALUE: String = "value"
  protected[avro] val RESPONSE: String = "response"
  protected[avro] val OFFSET: String = "offset"
  protected[avro] val WEIGHT: String = "weight"
  protected[avro] val UID: String = "uid"
}
