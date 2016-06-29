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
package com.linkedin.photon.ml.algorithm

import com.linkedin.photon.ml.constants.StorageLevel
import com.linkedin.photon.ml.data.{FixedEffectDataSet, GameDatum, KeyValueScore, LabeledPoint}
import com.linkedin.photon.ml.evaluation.Evaluator
import com.linkedin.photon.ml.function.DiffFunction
import com.linkedin.photon.ml.optimization.game.OptimizationTracker
import com.linkedin.photon.ml.model.{DatumScoringModel, GAMEModel, FixedEffectModel}
import com.linkedin.photon.ml.normalization.NoNormalization
import com.linkedin.photon.ml.optimization.{GeneralizedLinearOptimizationProblem, OptimizationStatesTracker}
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import com.linkedin.photon.ml.util.PhotonLogger

import breeze.linalg.Vector
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.mockito.Matchers
import org.mockito.Mockito._
import org.testng.Assert._
import org.testng.annotations.{DataProvider, Test}

class FixedEffectCoordinateTest {

  @Test
  def testUpdateModel(): Unit = {
    // Create mocks
    val dataset = mock(classOf[FixedEffectDataSet])
    val optimizationProblem = mock(
      classOf[GeneralizedLinearOptimizationProblem[GeneralizedLinearModel, DiffFunction[LabeledPoint]]])

    val fixedEffectModel = mock(classOf[FixedEffectModel])
    val model = mock(classOf[GeneralizedLinearModel])
    val updatedModel = mock(classOf[GeneralizedLinearModel])
    val labeledPoints = mock(classOf[RDD[(Long, LabeledPoint)]])
    val labeledPointValues = mock(classOf[RDD[LabeledPoint]])
    val sparkContext = mock(classOf[SparkContext])
    val modelBroadcast = mock(classOf[Broadcast[GeneralizedLinearModel]])
    val statesTracker = mock(classOf[OptimizationStatesTracker])

    // Optimization problem
    doReturn(labeledPoints).when(optimizationProblem).downSample(
      Matchers.eq(labeledPoints), Matchers.any(classOf[Long]))
    doReturn(Some(statesTracker)).when(optimizationProblem).getStatesTracker
    doReturn(updatedModel).when(optimizationProblem).run(labeledPointValues, model, NoNormalization)

    // Fixed effect model
    doReturn(model).when(fixedEffectModel).model
    doReturn(fixedEffectModel).when(fixedEffectModel).update(modelBroadcast)

    // Dataset
    doReturn(labeledPoints).when(dataset).labeledPoints
    doReturn(sparkContext).when(dataset).sparkContext

    // Spark context
    doReturn(modelBroadcast).when(sparkContext).broadcast(updatedModel)

    // Labeled points
    doReturn(labeledPoints).when(labeledPoints).setName(Matchers.any(classOf[String]))
    doReturn(labeledPoints).when(labeledPoints).persist(Matchers.any())
    doReturn(labeledPointValues).when(labeledPoints).values

    // Update model
    val coordinate = new FixedEffectCoordinate(dataset, optimizationProblem)
    val (resultModel, tracker) = coordinate.updateModel(fixedEffectModel)

    verify(optimizationProblem, times(1)).run(
      Matchers.any(classOf[RDD[LabeledPoint]]), Matchers.eq(model), Matchers.eq(NoNormalization))
  }
}
