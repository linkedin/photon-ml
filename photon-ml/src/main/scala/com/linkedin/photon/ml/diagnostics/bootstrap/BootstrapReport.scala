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
