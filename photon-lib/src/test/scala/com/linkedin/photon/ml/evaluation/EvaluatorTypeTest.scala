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
package com.linkedin.photon.ml.evaluation

import org.testng.Assert._
import org.testng.annotations.Test

/**
 * Unit tests for [[EvaluatorType]].
 */
class EvaluatorTypeTest {

  /**
   * Test that better evaluation metrics can be distinguished from worse ones for each [[EvaluatorType]].
   */
  @Test
  def testBetterThan(): Unit = {

    // AUC
    assertFalse(EvaluatorType.AUC.betterThan(0D, 1D))
    assertFalse(EvaluatorType.AUC.betterThan(1D, 1D))
    assertTrue(EvaluatorType.AUC.betterThan(1D, 0D))

    // AUPR
    assertFalse(EvaluatorType.AUPR.betterThan(0D, 1D))
    assertFalse(EvaluatorType.AUPR.betterThan(1D, 1D))
    assertTrue(EvaluatorType.AUPR.betterThan(1D, 0D))

    // RMSE
    assertTrue(EvaluatorType.RMSE.betterThan(0D, 1D))
    assertFalse(EvaluatorType.RMSE.betterThan(1D, 1D))
    assertFalse(EvaluatorType.RMSE.betterThan(1D, 0D))

    // Logistic Loss
    assertTrue(EvaluatorType.LogisticLoss.betterThan(0D, 1D))
    assertFalse(EvaluatorType.LogisticLoss.betterThan(1D, 1D))
    assertFalse(EvaluatorType.LogisticLoss.betterThan(1D, 0D))

    // Poisson Loss
    assertTrue(EvaluatorType.PoissonLoss.betterThan(0D, 1D))
    assertFalse(EvaluatorType.PoissonLoss.betterThan(1D, 1D))
    assertFalse(EvaluatorType.PoissonLoss.betterThan(1D, 0D))

    // Squared Loss
    assertTrue(EvaluatorType.SquaredLoss.betterThan(0D, 1D))
    assertFalse(EvaluatorType.SquaredLoss.betterThan(1D, 1D))
    assertFalse(EvaluatorType.SquaredLoss.betterThan(1D, 0D))
  }
}
