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
package com.linkedin.photon.ml.diagnostics.fitting

import com.linkedin.photon.ml.{DataValidationType, Evaluation, ModelTraining}
import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.normalization.NoNormalization
import com.linkedin.photon.ml.optimization.{L2RegularizationContext, OptimizerType}
import com.linkedin.photon.ml.supervised.TaskType
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import com.linkedin.photon.ml.test.SparkTestUtils
import org.apache.spark.rdd.RDD
import org.testng.annotations.Test

/**
 * Integration tests for FittingDiagnostic
 */
class FittingDiagnosticIntegTest extends SparkTestUtils {
  import FittingDiagnosticIntegTest._
  import org.testng.Assert._

  /**
   * Happy path
   */
  @Test
  def checkHappyPath():Unit = sparkTest("checkHappyPath") {
    val data = sc.parallelize(drawBalancedSampleFromNumericallyBenignDenseFeaturesForBinaryClassifierLocal(SEED, SIZE, DIMENSION).map( x => new LabeledPoint(x._1, x._2)).toSeq).repartition(4).cache

    val modelFit = (data:RDD[LabeledPoint], warmStart:Map[Double, GeneralizedLinearModel]) => {
      ModelTraining.trainGeneralizedLinearModel(
        data,
        TaskType.LOGISTIC_REGRESSION,
        OptimizerType.TRON,
        L2RegularizationContext,
        LAMBDAS,
        NoNormalization,
        NUM_ITERATIONS,
        TOLERANCE,
        false,
        None,
        warmStart,
        1)._1
    }

    val diagnostic = new FittingDiagnostic()

    val reports = diagnostic.diagnose(modelFit, data, None)

    assertFalse(reports.isEmpty)
    assertEquals(reports.size, LAMBDAS.length)
    LAMBDAS.map(lambda => assertTrue(reports.contains(lambda), s"Report contains lambda $lambda"))

    reports.map(x => {
      val (_, report) = x

      assertFalse(report.metrics.isEmpty)
      val expectedKeys = List(Evaluation.ROOT_MEAN_SQUARE_ERROR, Evaluation.MEAN_ABSOLUTE_ERROR, Evaluation.MEAN_SQUARE_ERROR)
      expectedKeys.map(y => {
        assertTrue(report.metrics.contains(y))
        assertEquals(report.metrics.get(y).get._1.size, FittingDiagnostic.NUM_TRAINING_PARTITIONS - 1)
        assertEquals(report.metrics.get(y).get._2.size, FittingDiagnostic.NUM_TRAINING_PARTITIONS - 1)
        assertEquals(report.metrics.get(y).get._3.size, FittingDiagnostic.NUM_TRAINING_PARTITIONS - 1)
        // Make sure that training set fraction is monotonically increasing
        assertTrue(report.metrics.get(y).get._1.foldLeft((0.0, true))((prev, current) => (current, prev._2 && current > prev._1))._2)
      })
    })
  }
}

object FittingDiagnosticIntegTest {
  val SEED = 0xdeadbeef
  val SIZE = 500000
  val DIMENSION = 100
  val NUM_ITERATIONS = 100
  val TOLERANCE = 1e-2
  val LAMBDAS = List(1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5)
}
