/*
 * Copyright 2018 LinkedIn Corp. All rights reserved.
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
package com.linkedin.photon.ml.hyperparameter

/**
 * Hyper-parameter default values
 */
object GameHyperparameterDefaults {

  val priorDefault: Map[String, String] = Map(
    "global_regularizer" -> "0.0",
    "member_regularizer" -> "0.0",
    "item_regularizer" -> "0.0")

  val configDefault: String =
    """
      |{ "tuning_mode" : "BAYESIAN",
      |  "variables" : {
      |    "global_regularizer" : {
      |      "type" : "FLOAT",
      |      "transform" : "LOG",
      |      "min" : -3,
      |      "max" : 3
      |    },
      |    "member_regularizer" : {
      |      "type" : "FLOAT",
      |      "transform" : "LOG",
      |      "min" : -3,
      |      "max" : 3
      |    },
      |    "item_regularizer" : {
      |      "type" : "FLOAT",
      |      "transform" : "LOG",
      |      "min" : -3,
      |      "max" : 3
      |    }
      |  }
      |}
    """.stripMargin
}
