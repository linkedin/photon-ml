package com.linkedin.photon.ml.diagnostics

import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.diagnostics.reporting.LogicalReport
import com.linkedin.photon.ml.stat.BasicStatisticalSummary
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import org.apache.spark.rdd.RDD

/**
 * General interface for model diagnostics. In this case, all we require is a particular signature. We will expand
 * these later with more concrete tests.
 *
 * Model diagnostics are diagnostics that are intended to give us information about a particular model given a
 * validation set. This is in contrast to training diagnostics, which are intended to tell us about a class of models
 * given a training set.
 *
 * @tparam M
 *           Input model type
 * @tparam D
 *           Output diagnostic type
 */
trait TrainingDiagnostic[-M <: GeneralizedLinearModel, +D <: LogicalReport] {
  /**
   * Compute training-time diagnostics.
   *
   * @param modelFactory
   *                     Functor that, given a data set, produces a set of (lambda, model) tuples
   *
   * @param trainingData
   *                     Set of <em>training</em> data
   *
   * @return
   *         A logical report encapsulating this diagnostic's findings
   */
  def diagnose(modelFactory:RDD[LabeledPoint]=>List[(Double, M)], trainingData:RDD[LabeledPoint], summary:Option[BasicStatisticalSummary]): Map[Double, D]
}
