package com.linkedin.photon.ml.diagnostics.independence

import com.linkedin.photon.ml.diagnostics.reporting.LogicalReport

/**
 * Analysis of independence of error and prediction.
 *
 * @param errorSample Sample of errors. This should have same length as [[predictionSample]]
 * @param predictionSample Sample of predictions. This should have the same length as [[errorSample]]
 * @param kentallTau Kendall &tau; independence test report
 */
case class PredictionErrorIndependenceReport(val errorSample: Array[Double],
                                             val predictionSample: Array[Double],
                                             val kentallTau: KendallTauReport) extends LogicalReport

