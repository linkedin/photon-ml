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
package com.linkedin.photon.ml.sampler

import scala.util.Random

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.testng.Assert
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.test.{CommonTestUtils, SparkTestUtils}

/**
 * Integration tests for [[DefaultDownSampler]].
 */
class DefaultDownSamplerIntegTest extends SparkTestUtils {

  private val numTimesToRun = 100
  private val numInstancesToGenerate = 100
  private val numFeatures = 1

  /**
   * Generates a random labeled point with given number of features having a random label, default offset (0.0)
   * and default weight (1.0).
   *
   * @param numFeatures The feature dimension of the dummy data
   * @return A labeled point
   */
  private def generateRandomLabeledPoint(numFeatures: Integer): LabeledPoint = {
    new LabeledPoint(new Random().nextDouble(), CommonTestUtils.generateDenseFeatureVectors(1, 0, numFeatures).head)
  }

  /**
   * Generate a dummy RDD[(Long, LabeledPoint)] for testing the DefaultDownSampler.
   *
   * @param sc The Spark context used to convert the list of points to and RDD
   * @param numInstances The number of training samples
   * @param numFeatures The feature dimension of the dummy data
   * @return An RDD of dummy training data
   */
  private def generateDummyDataset(
      sc: SparkContext,
      numInstances: Integer,
      numFeatures: Integer): RDD[(Long, LabeledPoint)] =
    sc.parallelize((0 until numInstances).map(i => (i.toLong, generateRandomLabeledPoint(numFeatures))))

  @DataProvider
  def validDownSamplingRatesProvider(): Array[Array[Any]] =
    Array(Array(0.25), Array(0.5), Array(0.75))

  @DataProvider
  def invalidDownSamplingRatesProvider(): Array[Array[Any]] =
    Array(Array(-0.5), Array(0.0), Array(1.0), Array(1.5))

  /**
   * Test that using the [[DefaultDownSampler]] generates a new [[RDD]] with an approximately correct number of
   * instances as per the down-sampling rate.
   *
   * Down sampling is run multiple times and number of instances in each run is accumulated to allow law of large
   * numbers to kick in.
   *
   * @param downSamplingRate The down-sampling rate
   */
  @Test(dataProvider = "validDownSamplingRatesProvider")
  def testDownSampling(downSamplingRate: Double): Unit = sparkTest("testDownSampling") {

    val dataSet = generateDummyDataset(sc, numInstancesToGenerate, numFeatures)

    var numInstancesInSampled: Long = 0
    for (_ <- 0 until numTimesToRun) {
      numInstancesInSampled += new DefaultDownSampler(downSamplingRate)
        .downSample(dataSet)
        .count()
    }

    val mean = numTimesToRun * numInstancesToGenerate * downSamplingRate
    val variance = mean * (1 - downSamplingRate)
    // tolerance = standard deviation * 5
    val tolerance = math.sqrt(variance) * 5

    Assert.assertEquals(numInstancesInSampled, mean, tolerance)
  }

  /**
   * Test that bad down-sampling rates will be rejected.
   *
   * @param downSamplingRate The down-sampling rate
   */
  @Test(dataProvider = "invalidDownSamplingRatesProvider", expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testBadRates(downSamplingRate: Double): Unit = sparkTest("testBadRates") {
    new DefaultDownSampler(downSamplingRate)
  }
}
