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
package com.linkedin.photon.ml.transformers

import org.apache.spark.SparkContext
import org.apache.spark.ml.param.ParamMap
import org.mockito.Mockito._
import org.slf4j.Logger
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.data.InputColumnsNames
import com.linkedin.photon.ml.evaluation.EvaluatorType.AUC
import com.linkedin.photon.ml.model.GameModel

/**
 * Unit tests for the [[GameTransformer]].
 */
class GameTransformerTest {

  /**
   * Test that a [[ParamMap]] with only required parameters set can be valid input.
   */
  @Test
  def testMinValidParamMap(): Unit = {

    val mockSparkContext = mock(classOf[SparkContext])
    val mockLogger = mock(classOf[Logger])
    val mockGameModel = mock(classOf[GameModel])

    val transformer = new GameTransformer(mockSparkContext, mockLogger)

    transformer.setModel(mockGameModel)

    transformer.validateParams()
  }

  /**
   * Test that a [[ParamMap]] with all parameters set can be valid input.
   */
  @Test
  def testMaxValidParamMap(): Unit = {

    val validationEvaluators = Seq(AUC)
    val logDataAndModelStats = true
    val spillScoresToDisk = true

    val mockSparkContext = mock(classOf[SparkContext])
    val mockLogger = mock(classOf[Logger])
    val mockGameModel = mock(classOf[GameModel])
    val mockInputColumnNames = mock(classOf[InputColumnsNames])

    val transformer = new GameTransformer(mockSparkContext, mockLogger)

    transformer.setModel(mockGameModel)
    transformer.setInputColumnNames(mockInputColumnNames)
    transformer.setValidationEvaluators(validationEvaluators)
    transformer.setLogDataAndModelStats(logDataAndModelStats)
    transformer.setSpillScoresToDisk(spillScoresToDisk)

    transformer.validateParams()
  }

  @DataProvider
  def invalidTransformers(): Array[Array[Any]] = {

    val mockSparkContext = mock(classOf[SparkContext])
    val mockLogger = mock(classOf[Logger])
    val mockGameModel = mock(classOf[GameModel])

    val transformer = new GameTransformer(mockSparkContext, mockLogger)

    transformer.setModel(mockGameModel)

    transformer.validateParams()

    var result = Seq[Array[Any]]()
    var badTransformer = transformer

    // No training task
    badTransformer = transformer.copy(ParamMap.empty)
    badTransformer.clear(badTransformer.model)
    result = result :+ Array[Any](badTransformer)

    result.toArray
  }

  /**
   * Test that invalid parameters will be correctly rejected.
   *
   * @param transformer A [[GameTransformer]] with one or more flaws in its parameters
   */
  @Test(dataProvider = "invalidTransformers", expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testValidateParams(transformer: GameTransformer): Unit = transformer.validateParams()

  /**
   * Test that default values are set for all parameters that require them.
   */
  @Test
  def testDefaultParams(): Unit = {

    val mockSparkContext = mock(classOf[SparkContext])
    val mockLogger = mock(classOf[Logger])

    val transformer = new GameTransformer(mockSparkContext, mockLogger)

    transformer.getOrDefault(transformer.inputColumnNames)
    transformer.getOrDefault(transformer.logDataAndModelStats)
    transformer.getOrDefault(transformer.spillScoresToDisk)
  }
}
