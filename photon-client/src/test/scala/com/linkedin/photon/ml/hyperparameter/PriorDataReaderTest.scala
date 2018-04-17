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
import org.testng.Assert.assertEquals
import org.testng.annotations.Test

/**
 * Unit tests for [[PriorDataReader]].
 */
class PriorDataReaderTest {

  /**
   * Test that prior observation data can be loaded from JSON.
   */
  @Test
  def testFromJson(): Unit = {

    val priorDataJson = "{\"records\":[" +
      "{\"alpha\": \"1.0\",\"lambda\": \"2.0\",\"gamma\": \"3.0\",\"evaluationValue\": \"2.0\"}," +
      "{\"alpha\": \"0.5\",\"evaluationValue\": \"-2.0\"}]}"
    val priorDefault: Map[String, String] = Map("alpha" -> "1.0", "lambda" -> "4.0", "gamma" -> "8.0")
    val hyperParameterList = Seq("alpha", "lambda", "gamma")

    val priorData = PriorDataReader.fromJson(priorDataJson, priorDefault, hyperParameterList)
    val trueData = Seq(
      (DenseVector(Array(1.0, 2.0 ,3.0)), 2.0),
      (DenseVector(Array(0.5, 4.0, 8.0)), -2.0)
    )

    assertEquals(priorData, trueData)
  }
}
