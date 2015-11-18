/*
 * Copyright 2015 LinkedIn Corp. All rights reserved.
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
package com.linkedin.photon.ml.diagnostics.fitting

import com.linkedin.photon.ml.{DataValidationType, Evaluation, ModelTraining}
import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.normalization.NoNormalization
import com.linkedin.photon.ml.optimization.{L2RegularizationContext, OptimizerType}
import com.linkedin.photon.ml.supervised.TaskType
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import com.linkedin.photon.ml.test.SparkTestUtils
import org.apache.spark.rdd.RDD

/**
 * Integration tests for FittingDiagnostic
 */
class FittingDiagnosticIntegTest extends SparkTestUtils {
  import FittingDiagnosticIntegTest._
  import org.testng.Assert._

  /**
   * Happy path
   */
  def checkHappyPath():Unit = sparkTest("checkHappyPath") {
    val data = sc.parallelize(drawBalancedSampleFromNumericallyBenignDenseFeaturesForBinaryClassifierLocal(SEED, SIZE, DIMENSION).map( x => new LabeledPoint(x._1, x._2)).toSeq)

    val modelFit = (data:RDD[LabeledPoint]) => {
      ModelTraining.trainGeneralizedLinearModel(
        data,
        TaskType.LOGISTIC_REGRESSION,
        OptimizerType.TRON,
        L2RegularizationContext,
        List(1.0),
        NoNormalization,
        NUM_ITERATIONS,
        TOLERANCE,
        false,
        DataValidationType.VALIDATE_DISABLED,
        None)._1
    }

    val diagnostic = new FittingDiagnostic()

    val reports = diagnostic.diagnose(modelFit, data, None)

    assertFalse(reports.isEmpty)
    assertEquals(reports.size, 1)
    assertTrue(reports.contains(1.0))

    val report = reports.get(1.0).get

    assertFalse(report.metrics.isEmpty)
    val expectedKeys = List(Evaluation.ROOT_MEAN_SQUARE_ERROR, Evaluation.MEAN_ABSOLUTE_ERROR, Evaluation.MEAN_SQUARE_ERROR)
    expectedKeys.map(x => {
      assertTrue(report.metrics.contains(x))
      assertEquals(report.metrics.get(x).get._1.size, FittingDiagnostic.NUM_TRAINING_PARTITIONS - 1)
      assertEquals(report.metrics.get(x).get._2.size, FittingDiagnostic.NUM_TRAINING_PARTITIONS - 1)
      assertEquals(report.metrics.get(x).get._3.size, FittingDiagnostic.NUM_TRAINING_PARTITIONS - 1)
      // Make sure that training set fraction is monotonically increasing
      assertTrue(report.metrics.get(x).get._1.foldLeft((0.0, true))((prev, current) => (current, prev._2 && current > prev._1))._2)
    })
  }
}

object FittingDiagnosticIntegTest {
  val SEED = 0
  val SIZE = 100000
  val DIMENSION = 100
  val NUM_ITERATIONS = 1000
  val TOLERANCE = 1e-5
}
