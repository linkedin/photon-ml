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
package com.linkedin.photon.ml.sampling

import org.apache.spark.rdd.RDD
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.Types.UniqueSampleId
import com.linkedin.photon.ml.data.LabeledPoint

/**
 * Unit tests for [[DownSampler]].
 */
class DownSamplerTest {

  import DownSamplerTest._

  @DataProvider
  def invalidDownSamplingRatesProvider(): Array[Array[Any]] =
    Array(Array(-0.5), Array(0.0), Array(1.0), Array(1.5))

  /**
   * Test that bad down-sampling rates will be rejected.
   *
   * @param downSamplingRate The down-sampling rate
   */
  @Test(
    dataProvider = "invalidDownSamplingRatesProvider",
    expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testBadRates(downSamplingRate: Double): Unit = new MockDownSampler(downSamplingRate)
}

object DownSamplerTest {

  /**
   * Mock [[DownSampler]] class used for above tests.
   *
   * @param downSamplingRate The down-sampling rate
   */
  private class MockDownSampler(override val downSamplingRate: Double) extends DownSampler {

    override def downSample(
        labeledPoints: RDD[(UniqueSampleId, LabeledPoint)],
        seed: Long = DownSampler.getSeed): RDD[(UniqueSampleId, LabeledPoint)] = null
  }
}
