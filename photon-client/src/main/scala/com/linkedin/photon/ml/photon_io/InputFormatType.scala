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
package com.linkedin.photon.ml.photon_io

/**
 * An enum type that indicates the input data file formats.
 *
 * Currently supports:
 *   AVRO: serialized format using Avro: https://avro.apache.org/
 *   LIBSVM: a text format following conventions indicated
 *       by LIBSVM: https://www.csie.ntu.edu.tw/~cjlin/libsvm/
 */
object InputFormatType extends Enumeration {
  type InputFormatType = Value
  val AVRO, LIBSVM, NONE = Value
}
