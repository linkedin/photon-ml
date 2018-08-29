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

import breeze.linalg.DenseVector
import com.linkedin.photon.ml.HyperparameterTuningMode
import org.testng.Assert.{assertEquals, assertNotEquals, assertTrue}
import org.testng.annotations.Test

/**
 * Unit tests for [[HyperparameterSerialization]].
 */
class HyperparameterSerializationTest {

  /**
   * Test that prior observation data can be loaded from JSON.
   */
  @Test
  def testPriorFromJson(): Unit = {

    val priorDataJson =
      """
        |{
        |  "records": [
        |    {
        |      "alpha": "1.0",
        |      "lambda": "2.0",
        |      "gamma": "3.0",
        |      "evaluationValue": "0.01"
        |    },
        |    {
        |      "alpha": "0.5",
        |      "evaluationValue": "0.02"
        |    }
        |  ]
        |}
      """.stripMargin
    val priorDefault: Map[String, String] = Map("alpha" -> "1.0", "lambda" -> "4.0", "gamma" -> "8.0")
    val hyperParameterList = Seq("alpha", "lambda", "gamma")

    val priorData = HyperparameterSerialization.priorFromJson(priorDataJson, priorDefault, hyperParameterList)
    val expectedData = Seq(
      (DenseVector(Array(1.0, 2.0 ,3.0)), 0.01),
      (DenseVector(Array(0.5, 4.0, 8.0)), 0.02)
    )

    assertEquals(priorData, expectedData)
  }

  /**
   * Unit test to set hyper-parameter configuration by default.
   */
  @Test
  def testConfigFromJson(): Unit = {

    val config: String =
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

    val hyperParams = HyperparameterSerialization.configFromJson(config)

    assertEquals(hyperParams.tuningMode, HyperparameterTuningMode.BAYESIAN)
    assertNotEquals(hyperParams.tuningMode, HyperparameterTuningMode.NONE)

    // Testing matching variables name
    assertEquals(hyperParams.names.toSet, Set("global_regularizer", "member_regularizer", "item_regularizer"))

    // Testing the corresponding ranges.
    val matchingRanges = hyperParams.ranges.zipWithIndex.map(
      row => (hyperParams.names(row._2), (row._1.start, row._1.end)))

    assertEquals(
      matchingRanges.toSet,
      Set(
        ("global_regularizer", (-3, 3)),
        ("member_regularizer", (-3, 3)),
        ("item_regularizer", (-3, 3))))

    assertTrue(hyperParams.discreteParams.isEmpty)

    // Testing the transformation Map.
    assertEquals(
      hyperParams.transformMap.toSet,
      Set(
        hyperParams.names.indexOf("global_regularizer") -> "LOG",
        hyperParams.names.indexOf("member_regularizer") -> "LOG",
        hyperParams.names.indexOf("item_regularizer") -> "LOG"))
  }
}
