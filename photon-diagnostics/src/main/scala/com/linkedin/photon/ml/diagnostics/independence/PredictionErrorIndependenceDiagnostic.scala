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
package com.linkedin.photon.ml.diagnostics.independence

import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.diagnostics.ModelDiagnostic
import com.linkedin.photon.ml.stat.FeatureDataStatistics
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel

/**
  * Perform several tests of independence to see if prediction errors and predictions are independent.
  */
class PredictionErrorIndependenceDiagnostic
  extends ModelDiagnostic[GeneralizedLinearModel, PredictionErrorIndependenceReport] {

  import PredictionErrorIndependenceDiagnostic._

  override def diagnose(
      model: GeneralizedLinearModel,
      data: RDD[LabeledPoint],
      summary: Option[FeatureDataStatistics]): PredictionErrorIndependenceReport = {

    val broadcastModel = data.sparkContext.broadcast(model)
    val predictionError = data.map { labeledPoint =>
      val prediction = broadcastModel.value.computeMeanFunctionWithOffset(labeledPoint.features, labeledPoint.offset)
      val error = labeledPoint.label - prediction
      (prediction, error)
    }
    broadcastModel.unpersist()

    val sample = predictionError.takeSample(WITH_REPLACEMENT, MAXIMUM_SAMPLE_SIZE)
    val predictionSamples = sample.map(_._1)
    val errorSamples = sample.map(_._2)
    val kendallTau = KENDALL_TAU_ANALYSIS.analyze(sample)
    new PredictionErrorIndependenceReport(errorSamples, predictionSamples, kendallTau)
  }
}

object PredictionErrorIndependenceDiagnostic {
  val WITH_REPLACEMENT = false
  val MAXIMUM_SAMPLE_SIZE = 5000
  val KENDALL_TAU_ANALYSIS = new KendallTauAnalysis
}
