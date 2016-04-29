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
import org.testng.annotations.{Test, DataProvider}

/**
 * Tests that using the BinaryClassificationDownSampler generates a new RDD with approximately correct number of
 * instances of each class as per the down-sampling rate. Also tests that the weights have been appropriately modified.
 *
 * Down sampling is run multiple times and number of instances in each run is accumulated to allow law of large
 * numbers to kick in
 *
 * @author nkatariy
 */
class BinaryClassificationDownSamplerIntegTest extends SparkTestUtils {
  val numTimesToRun = 100

  val numPositivesToGenerate = 10
  val numNegativesToGenerate = 100
  val numFeatures = 5

  val tolerance = math.min(100.0 / numTimesToRun / numNegativesToGenerate, 1)

  /**
   * Generates a random labeled point with label 1.0 if isPositive is true and 0.0 otherwise. The offset and weight
   * take their default values which are 0.0 and 1.0 respectively.
   *
   * @param isPositive whether generated labeled point should belong to positive class
   * @return labeled point
   */
  private def generateRandomLabeledPoint(isPositive: Boolean, numFeatures: Integer): LabeledPoint = {
    new LabeledPoint(if (isPositive) 1.0 else 0.0, CommonTestUtils.generateDenseFeatureVectors(1, 0, numFeatures).head)
  }

  /**
   * Generate a dummy RDD[(Long, LabeledPoint)] for testing the BinaryClassificationDownSampler
   *
   * @param numPositives number of positives in the dataset
   * @param numNegatives number of negatives in the dataset
   * @param numFeatures number of features in the generated examples
   *
   * @return RDD
   */
  private def generateDummyDataset(sc: SparkContext, numPositives: Integer,
                                            numNegatives: Integer, numFeatures: Integer): RDD[(Long, LabeledPoint)] = {
    val pos = (0 until numPositives).map(i => (i.toLong, generateRandomLabeledPoint(isPositive = true, numFeatures)))
    val neg = (0 until numNegatives).map(i => (i.toLong, generateRandomLabeledPoint(isPositive = false, numFeatures)))
    val points: Seq[(Long, LabeledPoint)] = (pos ++ neg).toSeq
    sc.parallelize(points)
  }

  @DataProvider
  def downSamplingRatesProvider(): Array[Array[Any]] = {
    Array(Array(0.0), Array(0.25), Array(0.5), Array(0.75), Array(1.0))
  }

  @Test(dataProvider = "downSamplingRatesProvider")
  def testDownSampling(downSamplingRate: Double): Unit = sparkTest("testDownSampling")  {
    val dataset = generateDummyDataset(sc, numPositivesToGenerate, numNegativesToGenerate, numFeatures)

    var numNegativesInSampled: Long = 0
    for (x <- 0 until numTimesToRun) {
      val sampled = new BinaryClassificationDownSampler(downSamplingRate).downSample(dataset)
      val pos = sampled.filter({
        case (_, point) => point.label >= MathConst.POSITIVE_RESPONSE_THRESHOLD
      })
      Assert.assertEquals(pos.count(), numPositivesToGenerate)
      pos.foreach({
        case (_, point) => Assert.assertEquals(point.weight, 1.0, MathConst.MEDIUM_PRECISION_TOLERANCE_THRESHOLD)
      })
      val neg = sampled.filter({
        case (_, point) => point.label < MathConst.POSITIVE_RESPONSE_THRESHOLD
      })
      numNegativesInSampled += neg.count()
      neg.foreach({
        case (_, point) =>
          Assert.assertEquals(point.weight,
            1.0 / downSamplingRate,
            MathConst.MEDIUM_PRECISION_TOLERANCE_THRESHOLD)
      })
    }

    if (downSamplingRate == 0.0) {
      Assert.assertEquals(numNegativesInSampled, 0)
    } else if (downSamplingRate == 1.0) {
      Assert.assertEquals(numNegativesInSampled, numTimesToRun * numNegativesToGenerate)
    } else {
      Assert.assertEquals(numNegativesInSampled * 1.0 / numTimesToRun / numNegativesToGenerate,
        downSamplingRate,
        tolerance)
    }
  }
}