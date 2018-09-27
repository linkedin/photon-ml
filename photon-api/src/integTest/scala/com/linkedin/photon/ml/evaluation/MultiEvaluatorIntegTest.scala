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

import org.apache.spark.rdd.RDD
import org.testng.Assert.assertEquals
import org.testng.annotations.Test

import com.linkedin.photon.ml.Types.UniqueSampleId
import com.linkedin.photon.ml.test.SparkTestUtils

/**
 * Integration test cases for the [[MultiEvaluator]]
 */
class MultiEvaluatorIntegTest extends SparkTestUtils {

  import MultiEvaluatorIntegTest._

  /**
   * Test that the [[MultiEvaluator]] will correctly group records by ID and pass them to a [[LocalEvaluator]] for
   * evaluation.
   */
  @Test
  def testEvaluate(): Unit = sparkTest("testEvaluateWithLabelAndWeight") {

    // Create the following input:
    //
    //  UID | REID | Score
    // --------------------
    //   1  |  1   |   2
    //   2  |  0   |   3
    //   3  |  1   |   4
    //   4  |  0   |   5
    //
    val numUIDs = 4
    val rawInput = (1 to numUIDs).map { v =>
      (v.toLong, (v % 2).toString, (v + 1).toDouble)
    }
    val ids = sc.parallelize(
      rawInput.map { case (uid, reid, _) =>
        (uid, reid)
      })
    val scores = sc.parallelize(
      rawInput.map { case (uid, _, score) =>
        (uid, (score, 1D, 1D))
      })

    // Scores are grouped by REID, the metric is sum(scores) ^ 2, and the evaluation should be the average of the metric
    // values.
    //
    // scores_0 = [2, 4], sum_0 = 3 + 5 = 8, metric_0 = 6 ^ 2 = 64
    // scores_1 = [1, 3], sum_1 = 2 + 4 = 6, metric_1 = 4 ^ 2 = 36
    // avg_metric = (36 + 64) / 2 = 100 / 2 = 50
    //
    val mockMultiEvaluator = new MockMultiEvaluator(ids)
    val expectedResult = 50D

    assertEquals(mockMultiEvaluator.evaluate(scores), expectedResult)
  }
}

object MultiEvaluatorIntegTest {

  class MockLocalEvaluator extends LocalEvaluator {
    override protected[ml] def evaluate(scoreLabelAndWeight: Array[(Double, Double, Double)]): Double =
      math.pow(scoreLabelAndWeight.map(_._1).sum, 2)
  }

  class MockMultiEvaluator(ids: RDD[(UniqueSampleId, String)])
    extends MultiEvaluator(new MockLocalEvaluator, ids) {

    val evaluatorType = EvaluatorType.AUC
  }
}
