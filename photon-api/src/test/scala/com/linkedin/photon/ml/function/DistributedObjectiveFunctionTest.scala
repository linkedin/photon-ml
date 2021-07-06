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

import org.mockito.Mockito._
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.function.glm.LogisticLossFunction
import com.linkedin.photon.ml.optimization.NoRegularizationContext
import com.linkedin.photon.ml.optimization.game.FixedEffectOptimizationConfiguration

/**
 * Tests for [[DistributedObjectiveFunction]]
 */
class DistributedObjectiveFunctionTest {

  import DistributedObjectiveFunctionTest._

  @DataProvider
  def invalidInput(): Array[Array[Any]] = Array(Array(-1), Array(0))

  /**
   * Test that [[DistributedObjectiveFunction]] will reject invalid input.
   *
   * @param treeAggregateDepth An invalid tree aggregate depth
   */
  @Test(dataProvider = "invalidInput", expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testSetupWithInvalidInput(treeAggregateDepth: Int): Unit =
    buildDistributedObjectiveFunction(treeAggregateDepth)
}

object DistributedObjectiveFunctionTest {

  val MOCK_REGULARIZATION_WEIGHT = 0D
  val MOCK_REGULARIZATION_CONTEXT = NoRegularizationContext
  val MOCK_COORDINATE_CONFIG = mock(classOf[FixedEffectOptimizationConfiguration])

  doReturn(MOCK_REGULARIZATION_WEIGHT).when(MOCK_COORDINATE_CONFIG).regularizationWeight
  doReturn(MOCK_REGULARIZATION_CONTEXT).when(MOCK_COORDINATE_CONFIG).regularizationContext

  /**
   * Helper function to build a [[DistributedObjectiveFunction]] object.
   *
   * @param treeAggregateDepth The tree aggregation depth (see [[DistributedObjectiveFunction]] for documentation)
   * @return A new [[DistributedObjectiveFunction]] object
   */
  def buildDistributedObjectiveFunction(treeAggregateDepth: Int): DistributedObjectiveFunction =
     DistributedObjectiveFunction(
       MOCK_COORDINATE_CONFIG,
       LogisticLossFunction,
       treeAggregateDepth,
       interceptIndexOpt = None)
}
