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
package com.linkedin.photon.ml.diagnostics.reporting.reports.model

import com.linkedin.photon.ml.diagnostics.bootstrap.BootstrapReport
import com.linkedin.photon.ml.diagnostics.featureimportance.FeatureImportanceReport
import com.linkedin.photon.ml.diagnostics.fitting.FittingReport
import com.linkedin.photon.ml.diagnostics.hl.HosmerLemeshowReport
import com.linkedin.photon.ml.diagnostics.independence.PredictionErrorIndependenceReport
import com.linkedin.photon.ml.diagnostics.reporting.LogicalReport
import com.linkedin.photon.ml.stat.BasicStatisticalSummary
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel

/**
 * Everything we might want to report about a model
 * @param model
 * The actual model that we fit
 * @param lambda
 * Regularization strength
 * @param modelDescription
 * Description of the model
 * @param nameIdxMap
 * Map of (encoded name/term &rarr; coefficient index)
 * @param metrics
 * Map of (metric name &rarr; metric value)
 * @param fitReport
 * Result of fitting training diagnostic
 * @param predictionErrorIndependence
 * Prediction / error independence analysis
 * @param hosmerLemeshow
 * Results of HL goodness-of-fit (only applicable for logistic regression)
 * @param meanImpactFeatureImportance
 * Feature importance as computed by
 *   [[com.linkedin.photon.ml.diagnostics.featureimportance.ExpectedMagnitudeFeatureImportanceDiagnostic]]
 * @param varianceImpactFeatureImportance
 * Feature importance as computed by
 *   [[com.linkedin.photon.ml.diagnostics.featureimportance.VarianceFeatureImportanceDiagnostic]]
 * @param bootstrapReport
 * Bootstrap training diagnostic report
 * @tparam GLM
 * Model type
 */
case class ModelDiagnosticReport[GLM <: GeneralizedLinearModel](
    val model: GLM,
    val lambda: Double,
    val modelDescription: String,
    val nameIdxMap: Map[String, Int],
    val metrics: Map[String, Double],
    val summary: Option[BasicStatisticalSummary],
    val predictionErrorIndependence: Option[PredictionErrorIndependenceReport],
    var hosmerLemeshow: Option[HosmerLemeshowReport],
    val meanImpactFeatureImportance: Option[FeatureImportanceReport],
    val varianceImpactFeatureImportance: Option[FeatureImportanceReport],
    val fitReport:Option[FittingReport],
    val bootstrapReport:Option[BootstrapReport])
  extends LogicalReport
