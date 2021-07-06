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

import org.testng.Assert.assertEquals
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.data.GameDatum
import com.linkedin.photon.ml.evaluation.EvaluatorType._
import com.linkedin.photon.ml.test.SparkTestUtils

/**
 * Integration tests for [[EvaluatorFactory]].
 */
class EvaluatorFactoryIntegTest extends SparkTestUtils {

  import EvaluatorFactoryIntegTest._

  /**
   * Provide [[EvaluatorType]]s.
   */
  @DataProvider
  def evaluatorTypeProvider(): Array[Array[Any]] =
    Array(
      Array(AUC),
      Array(RMSE),
      Array(PoissonLoss),
      Array(LogisticLoss),
      Array(SquaredLoss),
      Array(MultiPrecisionAtK(1, ID_TAG)),
      Array(MultiPrecisionAtK(5, ID_TAG)),
      Array(MultiAUC(ID_TAG)))

  /**
   * Test that the [[EvaluatorFactory]] can correctly construct [[Evaluator]]s from [[EvaluatorType]]s.
   */
  @Test(dataProvider = "evaluatorTypeProvider")
  def testBuildEvaluator(evaluatorType: EvaluatorType): Unit = sparkTest("testBuildEvaluator") {

    val gameDatum = new GameDatum(
      response = 1.0,
      offsetOpt = Some(0.0),
      weightOpt = None,
      featureShardContainer = Map(),
      idTagToValueMap = Map(ID_TAG -> "id"))
    val gameDataset = sc.parallelize(Seq((1L, gameDatum)))
    val evaluator = EvaluatorFactory.buildEvaluator(evaluatorType, gameDataset)

    assertEquals(evaluator.evaluatorType, evaluatorType)
  }
}

object EvaluatorFactoryIntegTest {

  private val ID_TAG = "idTag"
}
