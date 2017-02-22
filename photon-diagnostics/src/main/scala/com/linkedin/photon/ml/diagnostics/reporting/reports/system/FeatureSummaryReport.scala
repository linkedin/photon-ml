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
package com.linkedin.photon.ml.diagnostics.reporting.reports.system

import com.linkedin.photon.ml.diagnostics.reporting.LogicalReport
import com.linkedin.photon.ml.stat.BasicStatisticalSummary

/**
 * Report describing a feature summary
 * @param nameToIndex
 *                    Eventually a map of (featureId &rarr; index); for now, a map of a string encoding (key/term) to
 *                    feature index.
 * @param summary
 *                Feature summary
 */
case class FeatureSummaryReport(
  nameToIndex: Map[String, Int],
  summary:BasicStatisticalSummary) extends LogicalReport
