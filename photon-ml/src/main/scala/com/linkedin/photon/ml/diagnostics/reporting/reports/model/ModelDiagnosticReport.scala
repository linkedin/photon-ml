package com.linkedin.photon.ml.diagnostics.reporting.reports.model

import com.linkedin.photon.ml.diagnostics.featureimportance.FeatureImportanceReport
import com.linkedin.photon.ml.diagnostics.hl.HosmerLemeshowReport
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
 * @param hosmerLemeshow
 * Results of HL goodness-of-fit (only applicable for logistic regression)
 * @param meanImpactFeatureImportance
 * Feature importance as computed by [[com.linkedin.photon.ml.diagnostics.featureimportance.ExpectedMagnitudeFeatureImportanceDiagnostic]]
 * @param varianceImpactFeatureImportance
 * Feature importance as computed by [[com.linkedin.photon.ml.diagnostics.featureimportance.VarianceFeatureImportanceDiagnostic]]
 * @tparam GLM
 * Model type
 */
case class ModelDiagnosticReport[GLM <: GeneralizedLinearModel](
                                                                 val model: GLM,
                                                                 val lambda: Double,
                                                                 val modelDescription: String,
                                                                 val nameIdxMap: Map[String, Int],
                                                                 val metrics:Map[String, Double],
                                                                 val summary: Option[BasicStatisticalSummary],
                                                                 var hosmerLemeshow: Option[HosmerLemeshowReport],
                                                                 val meanImpactFeatureImportance:FeatureImportanceReport,
                                                                 val varianceImpactFeatureImportance:FeatureImportanceReport) extends LogicalReport
