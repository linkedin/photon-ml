/*
 * Copyright 2016 LinkedIn Corp. All rights reserved.
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
package com.linkedin.photon.ml.sampler

import com.linkedin.photon.ml.constants.MathConst
import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.test.{CommonTestUtils, SparkTestUtils}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.testng.Assert
import org.testng.annotations.{DataProvider, Test}

import scala.util.Random

/**
 * Tests that using the DefaultDownSampler generates a new RDD with approximately correct number of instances as per
 * the down-sampling rate
 *
 * Down sampling is run multiple times and number of instances in each run is accumulated to allow law of large
 * numbers to kick in
 *
 * @author nkatariy
 */
class DefaultDownSamplerIntegTest extends SparkTestUtils {
  val numTimesToRun = 100

  val numInstancesToGenerate = 100
  val numFeatures = 5

  val tolerance = math.min(100.0 / numTimesToRun / numInstancesToGenerate, 1.0)

  /**
   * Generates a random labeled point with given number of features having a random label, default offset (0.0)
   * and default weight (1.0)
   *
   * @return labeled point
   */
  private def generateRandomLabeledPoint(numFeatures: Integer): LabeledPoint = {
    new LabeledPoint(new Random().nextDouble(), CommonTestUtils.generateDenseFeatureVectors(1, 0, numFeatures).head)
  }

  /**
   * Generate a dummy RDD[(Long, LabeledPoint)] for testing the DefaultDownSampler
   *
   * @param numFeatures number of features in the generated examples
   *
   * @return RDD
   */
  private def generateDummyDataset(sc: SparkContext, numInstances: Integer,
                                   numFeatures: Integer): RDD[(Long, LabeledPoint)] = {
    sc.parallelize((0 until numInstances).map(i => (i.toLong, generateRandomLabeledPoint(numFeatures))))
  }

  @DataProvider
  def downSamplingRatesProvider(): Array[Array[Any]] = {
    Array(Array(0.0), Array(0.25), Array(0.5), Array(0.75), Array(1.0))
  }

  @Test(dataProvider = "downSamplingRatesProvider")
  def testDownSampling(downSamplingRate: Double): Unit = sparkTest("testDownSampling")  {
    val dataset = generateDummyDataset(
      sc,
      numInstancesToGenerate,
      numFeatures)

    var numInstancesInSampled: Long = 0
    for (x <- 0 until numTimesToRun) {
      numInstancesInSampled += new DefaultDownSampler(downSamplingRate)
        .downSample(dataset)
        .count()
    }

    if (downSamplingRate == 0.0) {
      Assert.assertEquals(numInstancesInSampled, 0)
    } else if (downSamplingRate == 1.0) {
      Assert.assertEquals(numInstancesInSampled, numTimesToRun * numInstancesToGenerate)
    } else {
      Assert.assertEquals(numInstancesInSampled * 1.0 / numTimesToRun / numInstancesToGenerate,
        downSamplingRate,
        tolerance)
    }
  }
}