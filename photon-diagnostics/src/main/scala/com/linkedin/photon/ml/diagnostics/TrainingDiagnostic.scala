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
 * Input model type
 * @tparam D
 * Output diagnostic type
 */
trait TrainingDiagnostic[-M <: GeneralizedLinearModel, +D <: LogicalReport] {
  /**
   * Compute training-time diagnostics without warm-start models.
   *
   * @param modelFactory
   * Functor that, given a data set, produces a set of (lambda, model) tuples
   *
   * @param trainingData
   * Set of <em>training</em> data
   *
   * @return
   * A logical report encapsulating this diagnostic's findings
   */
  def diagnose(
      modelFactory: (RDD[LabeledPoint], Map[Double, GeneralizedLinearModel]) => List[(Double, M)],
      trainingData: RDD[LabeledPoint],
      summary: Option[BasicStatisticalSummary]): Map[Double, D] =
    diagnose(modelFactory, Map.empty, trainingData, summary)

  /**
   * Compute training-time diagnostics, with (potentially empty) warm start models.
   *
   * @param modelFactory
   * Functor that, given a data set, produces a set of (lambda, model) tuples
   *
   * @param trainingData
   * Set of <em>training</em> data
   *
   * @return
   * A logical report encapsulating this diagnostic's findings
   */
  def diagnose(
      modelFactory: (RDD[LabeledPoint], Map[Double, GeneralizedLinearModel]) => List[(Double, M)],
      models:Map[Double, GeneralizedLinearModel],
      trainingData: RDD[LabeledPoint],
      summary: Option[BasicStatisticalSummary],
      seed: Long = System.nanoTime): Map[Double, D]
}
