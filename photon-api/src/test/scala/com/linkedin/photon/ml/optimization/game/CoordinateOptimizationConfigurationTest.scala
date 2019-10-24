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
package com.linkedin.photon.ml.optimization.game

import org.mockito.Mockito._
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.optimization.{OptimizerConfig, RegularizationContext}
import com.linkedin.photon.ml.util.DoubleRange

class CoordinateOptimizationConfigurationTest {

  @DataProvider
  def invalidInput(): Array[Array[Any]] = Array(
    Array(-1D, 1D, None, None, None),
    Array(1D, -1D, None, None, None),
    Array(1D, 0D, None, None, None),
    Array(1D, 2D, None, None, None),
    Array(1D, 1D, Some(DoubleRange(-1D, 10D)), None, None),
    Array(1D, 1D, None, Some(DoubleRange(0D, 1.1)), None),
    Array(-1D, 1D, None, None, Some(-1D)))

  /**
   * Test that [[FixedEffectOptimizationConfiguration]] will reject invalid input.
   *
   * @param regularizationWeight An invalid regularization weight
   * @param downSamplingRate An invalid down sampling rate
   */
  @Test(dataProvider = "invalidInput", expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testFixedEffectOptConfigSetupWithInvalidInput(
      regularizationWeight: Double,
      downSamplingRate: Double,
      regularizationWeightRange: Option[DoubleRange],
      elasticNetParamRange: Option[DoubleRange],
      incrementalWeight: Option[Double]): Unit = {

    val mockOptimizerConfig = mock(classOf[OptimizerConfig])
    val mockRegularizationContext = mock(classOf[RegularizationContext])

    FixedEffectOptimizationConfiguration(
      mockOptimizerConfig,
      mockRegularizationContext,
      regularizationWeight,
      regularizationWeightRange,
      elasticNetParamRange,
      incrementalWeight,
      downSamplingRate)
  }
}
