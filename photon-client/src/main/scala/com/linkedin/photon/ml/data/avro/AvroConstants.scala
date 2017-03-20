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
 * Constants used in Avro related functions and objects only.
 */
object AvroConstants {

  val DEFAULT_AVRO_FILE_NAME = "part-00000.avro"
  val ID_INFO = "id-info"
  val COEFFICIENTS = "coefficients"
  val FIXED_EFFECT = "fixed-effect"
  val RANDOM_EFFECT = "random-effect"
}
