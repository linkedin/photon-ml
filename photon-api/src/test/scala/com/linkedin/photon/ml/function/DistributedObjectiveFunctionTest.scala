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
package com.linkedin.photon.ml.function

import org.apache.spark.SparkContext
import org.mockito.Mockito._
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.normalization.NormalizationContext
import com.linkedin.photon.ml.util.BroadcastWrapper

/**
 * Tests for [[DistributedObjectiveFunction]]
 */
class DistributedObjectiveFunctionTest {

  import DistributedObjectiveFunctionTest.MockDistributedObjectiveFunction

  @DataProvider
  def invalidInput(): Array[Array[Any]] = Array(Array(-1), Array(0))

  /**
   * Test that [[DistributedObjectiveFunction]] will reject invalid input.
   *
   * @param treeAggregateDepth An invalid tree aggregate depth
   */
  @Test(dataProvider = "invalidInput", expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testSetupWithInvalidInput(treeAggregateDepth: Int): Unit = {

    val mockSparkContext = mock(classOf[SparkContext])

    new MockDistributedObjectiveFunction(mockSparkContext, treeAggregateDepth)
  }
}

object DistributedObjectiveFunctionTest {

  class MockDistributedObjectiveFunction(sc: SparkContext, treeAggregateDepth: Int)
    extends DistributedObjectiveFunction(sc, treeAggregateDepth) {

    override protected[ml] def value(
        input: Data,
        coefficients: Coefficients,
        normalizationContext: BroadcastWrapper[NormalizationContext]): Double = 0D
  }
}
