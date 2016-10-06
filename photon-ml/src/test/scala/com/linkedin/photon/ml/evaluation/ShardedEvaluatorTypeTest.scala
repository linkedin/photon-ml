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
package com.linkedin.photon.ml.evaluation

import org.testng.Assert._
import org.testng.annotations.{DataProvider, Test}

class ShardedEvaluatorTypeTest {

  @Test
  def testWithName(): Unit = {
    val precisionAt10 = " prEcIsiON@10:queryId   "
    assertEquals(ShardedPrecisionAtK(10, "queryId"), EvaluatorType.withName(precisionAt10))

    val auc = "   AuC:foobar "
    assertEquals(ShardedAUC("foobar"), EvaluatorType.withName(auc))
  }

  @DataProvider
  def generateUnrecognizedEvaluators(): Array[Array[Object]] = {
    Array(
      Array("precision"),
      Array("precision@"),
      Array("precision@k"),
      Array("precision@1k"),
      Array("precision@10"),
      Array("precision@queryId"),
      Array("precision@10queryId"),
      Array("precision@10|queryId"),
      Array("precision@10-queryId"),
      Array("auc@queryId"),
      Array("auc-queryId")
    )
  }

  @Test(dataProvider = "generateUnrecognizedEvaluators",
    expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testUnrecognizedEvaluatorsWithName(name: String): Unit = {
    EvaluatorType.withName(name)
  }
}
