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

class EvaluatorTypeTest {

  @Test
  def testWithName(): Unit = {

    val auc = "aUc"
    assertEquals(AUC, EvaluatorType.withName(auc))

    val rmse = "RMsE"
    assertEquals(RMSE, EvaluatorType.withName(rmse))

    val logisticLoss1 = "lOGIstiClosS"
    assertEquals(LogisticLoss, EvaluatorType.withName(logisticLoss1))
    val logisticLoss2 = "logiSTIC_LoSS"
    assertEquals(LogisticLoss, EvaluatorType.withName(logisticLoss2))

    val poissonLoss1 = "PoISSonLoSs"
    assertEquals(PoissonLoss, EvaluatorType.withName(poissonLoss1))
    val poissonLoss2 = "pOISson_lOSS"
    assertEquals(PoissonLoss, EvaluatorType.withName(poissonLoss2))

    val smoothedHingeLoss1 = "  sMooThEDHingELoss"
    assertEquals(SmoothedHingeLoss, EvaluatorType.withName(smoothedHingeLoss1))
    val smoothedHingeLoss2 = "SmOOTheD_Hinge_LOSS"
    assertEquals(SmoothedHingeLoss, EvaluatorType.withName(smoothedHingeLoss2))

    val squareLoss1 = "sQUAREDlosS "
    assertEquals(SquaredLoss, EvaluatorType.withName(squareLoss1))
    val squareLoss2 = "SquAREd_LOss"
    assertEquals(SquaredLoss, EvaluatorType.withName(squareLoss2))

    val precisionAt10 = " prEcIsiON@10:queryId   "
    assertEquals(PrecisionAtK(10, "queryId"), EvaluatorType.withName(precisionAt10))
  }

  @DataProvider
  def generateUnrecognizedEvaluators(): Array[Array[Object]] = {
    Array(
      Array("AreaUnderROCCurve"),
      Array("ROC"),
      Array("MSE"),
      Array("RRMSE"),
      Array("logistic"),
      Array("poisson"),
      Array("SVM"),
      Array("squared"),
      Array("121"),
      Array("null"),
      Array("precision"),
      Array("precision@"),
      Array("precision@k"),
      Array("precision@1k"),
      Array("precision@10"),
      Array("precision@queryId"),
      Array("precision@10queryId"),
      Array("precision@10|queryId"),
      Array("precision@10-queryId")
    )
  }

  @Test(dataProvider = "generateUnrecognizedEvaluators",
    expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testUnrecognizedEvaluatorsWithName(name: String): Unit = {
    EvaluatorType.withName(name)
  }
}
