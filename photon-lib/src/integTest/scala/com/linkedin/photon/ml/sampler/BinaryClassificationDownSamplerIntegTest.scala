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

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.testng.Assert.assertEquals
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.constants.MathConst
import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.test.{CommonTestUtils, SparkTestUtils}

/**
 * Integration tests for [[BinaryClassificationDownSampler]].
 */
class BinaryClassificationDownSamplerIntegTest extends SparkTestUtils {

  import BinaryClassificationDownSamplerIntegTest._

  @DataProvider
  def validDownSamplingRatesProvider(): Array[Array[Any]] = {
    Array(Array(0.25), Array(0.5), Array(0.75))
  }

  @DataProvider
  def invalidDownSamplingRatesProvider(): Array[Array[Any]] = {
    Array(Array(-0.5), Array(0.0), Array(1.0), Array(1.5))
  }

  /**
   * Test that using the [[BinaryClassificationDownSampler]] generates a new [[RDD]] with an approximately correct
   * number of instances of each class as per the down-sampling rate. Also tests that the weights have been
   * appropriately modified.
   *
   * Down sampling is run multiple times and number of instances in each run is accumulated to allow law of large
   * numbers to kick in.
   *
   * @param downSamplingRate The down-sampling rate
   */
  @Test(dataProvider = "validDownSamplingRatesProvider")
  def testDownSampling(downSamplingRate: Double): Unit = sparkTest("testDownSampling") {
    val dataset = generateDummyDataset(sc, NUM_POSITIVES_TO_GENERATE, NUM_NEGATIVES_TO_GENERATE, NUM_FEATURES)
    var numNegativesInSampled: Long = 0

    for (_ <- 0 until NUM_TIMES_TO_RUN) {
      val sampled = new BinaryClassificationDownSampler(downSamplingRate).downSample(dataset)

      // Need to burn seeds - the ones we get without this line cause failures in Travis CI
      DownSampler.getSeed

      val pos = sampled.filter { case (_, point) =>
        point.label >= MathConst.POSITIVE_RESPONSE_THRESHOLD
      }
      val neg = sampled.filter { case (_, point) =>
        point.label < MathConst.POSITIVE_RESPONSE_THRESHOLD
      }
      numNegativesInSampled += neg.count()

      assertEquals(pos.count(), NUM_POSITIVES_TO_GENERATE)
      pos.foreach { case (_, point) =>
        assertEquals(point.weight, 1.0, CommonTestUtils.LOW_PRECISION_TOLERANCE)
      }
      neg.foreach{ case (_, point) =>
        assertEquals(point.weight, 1.0 / downSamplingRate, CommonTestUtils.LOW_PRECISION_TOLERANCE)
      }
    }

    assertEquals(
      numNegativesInSampled * 1.0 / (NUM_TIMES_TO_RUN * NUM_NEGATIVES_TO_GENERATE),
      downSamplingRate,
      TOLERANCE)
  }

  /**
   * Test that bad down-sampling rates will be rejected.
   *
   * @param downSamplingRate The down-sampling rate
   */
  @Test(dataProvider = "invalidDownSamplingRatesProvider", expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testBadRates(downSamplingRate: Double): Unit = sparkTest("testBadRates") {
    new BinaryClassificationDownSampler(downSamplingRate)
  }
}

object BinaryClassificationDownSamplerIntegTest {

  private val NUM_TIMES_TO_RUN = 100
  private val NUM_POSITIVES_TO_GENERATE = 10
  private val NUM_NEGATIVES_TO_GENERATE = 100
  private val NUM_FEATURES = 5
  private val TOLERANCE = math.min(100.0 / (NUM_TIMES_TO_RUN * NUM_NEGATIVES_TO_GENERATE), 1)

  /**
   * Generates a random labeled point with label 1.0 if isPositive is true and 0.0 otherwise. The offset and weight
   * take their default values which are 0.0 and 1.0 respectively.
   *
   * @param isPositive Whether generated labeled point should belong to positive class
   * @param numFeatures The feature dimension of the dummy data
   * @return A labeled point
   */
  private def generateRandomLabeledPoint(isPositive: Boolean, numFeatures: Integer): LabeledPoint =
    new LabeledPoint(if (isPositive) 1.0 else 0.0, CommonTestUtils.generateDenseFeatureVectors(1, 0, numFeatures).head)

  /**
   * Generate a dummy RDD[(Long, LabeledPoint)] for testing the BinaryClassificationDownSampler.
   *
   * @param sc The Spark context used to convert the list of points to and RDD
   * @param numPositives The number of positives in the dataset
   * @param numNegatives The number of negatives in the dataset
   * @param numFeatures The feature dimension of the dummy data
   * @return An RDD of dummy training data
   */
  private def generateDummyDataset(
    sc: SparkContext,
    numPositives: Integer,
    numNegatives: Integer,
    numFeatures: Integer): RDD[(Long, LabeledPoint)] = {

    val pos = (0 until numPositives).map(i => (i.toLong, generateRandomLabeledPoint(isPositive = true, numFeatures)))
    val neg = (0 until numNegatives).map(i => (i.toLong, generateRandomLabeledPoint(isPositive = false, numFeatures)))
    val points: Seq[(Long, LabeledPoint)] = pos ++ neg
    sc.parallelize(points)
  }
}
