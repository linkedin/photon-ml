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
package com.linkedin.photon.ml.io

import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel

/**
 * Field names of the Avro formatted file used as input of [[GeneralizedLinearModel]]
 * @author xazhang
 */
trait FieldNames extends Serializable {
  val features: String
  val name: String
  val term: String
  val value: String
  val response: String
  val offset: String
  val weight: String
}