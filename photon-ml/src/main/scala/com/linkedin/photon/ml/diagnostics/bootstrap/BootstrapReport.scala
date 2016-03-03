package com.linkedin.photon.ml.diagnostics.bootstrap

import com.linkedin.photon.ml.diagnostics.reporting.LogicalReport
import com.linkedin.photon.ml.supervised.model.CoefficientSummary

/**
 * Everything we know as the result of a bootstrap diagnostic
 *
 * @param metricDistributions Map of metric &rarr; (min, q1, median, q3, max) value for that metric
 * @param bootstrappedModelMetrics Map of metric &rarr; value for the bagged / bootstrapped model (simple averaging)
 * @param importantFeatureCoefficientDistributions Map of (name, term) &rarr; coefficient summary
 */
case class BootstrapReport(
    metricDistributions: Map[String, (Double, Double, Double, Double, Double)],
    bootstrappedModelMetrics: Map[String, Double],
    importantFeatureCoefficientDistributions: Map[(String, String), CoefficientSummary],
    zeroCrossingFeatures: Map[(String, String), (Int, Double, CoefficientSummary)])
  extends LogicalReport
