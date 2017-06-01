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

import com.linkedin.photon.ml.evaluation.EvaluatorType.{RMSE, AUC}

/**
 * Integration test cases for the [[MultiEvaluatorType]]
 */
class MultiEvaluatorTypeTest {

  /**
   * Test that the [[MultiEvaluatorType]] correctly computes the set of unique ID tags for a group of [[EvaluatorType]]s.
   */
  @Test
  def testGetMultiEvaluatorIdTags(): Unit = {

    val expectedIdTagSet = Set("documentId", "queryId", "foo", "bar")
    val precisionAtKEvaluators = Set("documentId", "foo")
      .toSeq
      .flatMap(t => Seq(1, 3, 5, 10).map(MultiPrecisionAtK(_, t)))
    val areaUnderROCCurveEvaluators = Set("queryId", "bar").toSeq.map(MultiAUC(_))
    val allEvaluators = Seq(AUC, RMSE) ++ precisionAtKEvaluators ++ areaUnderROCCurveEvaluators

    assertEquals(MultiEvaluatorType.getMultiEvaluatorIdTags(allEvaluators), expectedIdTagSet)
  }
}
