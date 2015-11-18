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
package com.linkedin.photon.ml.optimization

import RegularizationType.RegularizationType
import org.testng.Assert
import org.testng.annotations.{DataProvider, Test}


/**
 * Test [[RegularizationContext]].
 *
 * @author dpeng
 */
class RegularizationContextTest {
  val epsilon = 1.0E-8
  @Test(dataProvider = "dataProvider")
  def testRegularizationContext(regularizationContext: RegularizationContext,
                                regularizationType: RegularizationType,
                                weight: Double,
                                l1Weight: Double,
                                l2Weight: Double): Unit = {
    Assert.assertEquals(regularizationContext.regularizationType, regularizationType)
    Assert.assertEquals(regularizationContext.getL1RegularizationWeight(weight), l1Weight, epsilon)
    Assert.assertEquals(regularizationContext.getL2RegularizationWeight(weight), l2Weight, epsilon)
  }

  @DataProvider(name = "dataProvider")
  def dataProvider(): Array[Array[Any]] = {
    val elastic1 = new RegularizationContext(RegularizationType.ELASTIC_NET, Some(0.5))
    val elastic2 = new RegularizationContext(RegularizationType.ELASTIC_NET, Some(0.1))
    Array(
      Array(L1RegularizationContext, RegularizationType.L1, 1.0d, 1.0d, 0.0d),
      Array(L2RegularizationContext, RegularizationType.L2, 1.0d, 0.0d, 1.0d),
      Array(elastic1, RegularizationType.ELASTIC_NET, 1.0d, 0.5d, 0.5d),
      Array(elastic2, RegularizationType.ELASTIC_NET, 2.0d, 0.2d, 1.8d)
    )
  }

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testTooLargeAlpha(): Unit = {
    new RegularizationContext(RegularizationType.ELASTIC_NET, Some(1.1d))
  }

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testNegativeAlpha(): Unit = {
    new RegularizationContext(RegularizationType.ELASTIC_NET, Some(-1d))
  }
}
