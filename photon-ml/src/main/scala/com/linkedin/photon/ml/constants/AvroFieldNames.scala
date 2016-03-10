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
package com.linkedin.photon.ml.constants

/**
 * Avro field names
 *
 * @author xazhang
 * @todo Unify the field names for different applications (feed, jymbii)
 */
object AvroFieldNames {
  val FEATURES: String = "features"
  val NAME: String = "name"
  val TERM: String = "term"
  val VALUE: String = "value"
  val RESPONSE: String = "response"
  val OFFSET: String = "offset"
  val WEIGHT: String = "weight"
  val EMPTY_STRING: String = ""
}
