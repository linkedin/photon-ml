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
package com.linkedin.photon.ml.cli.game

import org.testng.Assert._
import org.testng.annotations.Test

import com.linkedin.photon.ml.evaluation.EvaluatorType.{RMSE, AUC}
import com.linkedin.photon.ml.evaluation.{ShardedAUC, ShardedPrecisionAtK}

class EvaluatorParamsTest {

  @Test
  def testGetShardedEvaluatorIdTypes(): Unit = {
    val expectedIdTypeSet = Set("documentId", "queryId", "foo", "bar")
    val shardedPrecisionAtKEvaluators = Set("documentId", "foo")
      .toSeq
      .flatMap(t => Seq(1, 3, 5, 10).map(ShardedPrecisionAtK(_, t)))
    val shardedAreaUnderROCCurveEvaluators = Set("queryId", "bar").toSeq.map(ShardedAUC(_))
    val allEvaluators = Seq(AUC, RMSE) ++ shardedPrecisionAtKEvaluators ++ shardedAreaUnderROCCurveEvaluators

    val evaluatorParamsMocker = new EvaluatorParams {}
    evaluatorParamsMocker.evaluatorTypes = allEvaluators
    assertEquals(evaluatorParamsMocker.getShardedEvaluatorIdTypes, expectedIdTypeSet)
  }
}
