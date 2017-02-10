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

import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.mockito.Mockito._
import org.testng.annotations.Test

import com.linkedin.photon.ml.data.{FixedEffectDataSet, LabeledPoint}
import com.linkedin.photon.ml.function.DistributedObjectiveFunction
import com.linkedin.photon.ml.model.FixedEffectModel
import com.linkedin.photon.ml.optimization.{DistributedOptimizationProblem, OptimizationStatesTracker}
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel

/**
 * Tests for the FixedEffectCoordinate implementation
 */
class FixedEffectCoordinateIntegTest {
  @Test
  def testUpdateModel(): Unit = {
    // Create mocks
    val dataset = mock(classOf[FixedEffectDataSet])
    val optimizationProblem = mock(classOf[DistributedOptimizationProblem[DistributedObjectiveFunction]])

    val fixedEffectModel = mock(classOf[FixedEffectModel])
    val model = mock(classOf[GeneralizedLinearModel])
    val updatedModel = mock(classOf[GeneralizedLinearModel])
    val labeledPoints = mock(classOf[RDD[(Long, LabeledPoint)]])
    val sparkContext = mock(classOf[SparkContext])
    val modelBroadcast = mock(classOf[Broadcast[GeneralizedLinearModel]])
    val statesTracker = mock(classOf[OptimizationStatesTracker])

    // Optimization problem
    doReturn(updatedModel).when(optimizationProblem).runWithSampling(labeledPoints, model)
    doReturn(Some(statesTracker)).when(optimizationProblem).getStatesTracker

    // Fixed effect model
    doReturn(model).when(fixedEffectModel).model

    // Dataset
    doReturn(labeledPoints).when(dataset).labeledPoints
    doReturn(sparkContext).when(dataset).sparkContext

    // Spark context
    doReturn(modelBroadcast).when(sparkContext).broadcast(updatedModel)

    // Update model
    val coordinate = new FixedEffectCoordinate(dataset, optimizationProblem)
    coordinate.updateModel(fixedEffectModel)

    verify(optimizationProblem, times(1)).runWithSampling(labeledPoints, model)
  }
}
