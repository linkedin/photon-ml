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

import java.util.Random

import org.mockito.Mockito._
import org.testng.Assert.assertEquals
import org.testng.annotations.Test

import com.linkedin.photon.ml.test.SparkTestUtils

/**
 *
 */
class ShardedEvaluatorTest extends SparkTestUtils {
  @Test
  def testEvaluateWithLabelAndWeight(): Unit = sparkTest("testEvaluateWithLabelAndWeight") {

    val expectedResult = 0.5
    // Can't use Mockito here because we need the evaluator to be serializable
    val localEvaluatorMock = new LocalEvaluator {
      override protected[ml] def evaluate(
        scoreLabelAndWeight: Array[(Double, Double, Double)]): Double = expectedResult
    }

    val uniqueIds = (1 to 3).map(_.toLong)
    val ids = sc.parallelize(uniqueIds.map(long => long -> long.toString))
    val random = new Random(1L)
    val labelAndOffsetAndWeights =
      sc.parallelize(uniqueIds.map(long => long -> (random.nextDouble(), random.nextDouble(), random.nextDouble())))
    val scoresAndLabelsAndWeights =
      labelAndOffsetAndWeights.mapValues { case (label, _, weight) => (random.nextDouble(), label, weight) }

    val shardedEvaluator = mock(classOf[ShardedEvaluator], CALLS_REAL_METHODS)
    doReturn(labelAndOffsetAndWeights).when(shardedEvaluator).labelAndOffsetAndWeights
    doReturn(localEvaluatorMock).when(shardedEvaluator).localEvaluator
    doReturn(ids).when(shardedEvaluator).ids

    assertEquals(shardedEvaluator.evaluateWithScoresAndLabelsAndWeights(scoresAndLabelsAndWeights), expectedResult)
  }
}
