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
package com.linkedin.photon.ml.diagnostics.featureimportance

import com.linkedin.photon.ml.diagnostics.reporting.LogicalReport

import scala.collection.SortedMap

/**
 * @param importanceType
 * The type of feature importance that is being represented
 * @param importanceDescription
 * Description of how importance is computed
 * @param featureImportance
 * Map of (name, term) &rarr; (feature index, feature importance, feature description)
 * @param rankToImportance
 * Map of importance percentile / rank &rarr; importance @ that rank
 */
case class FeatureImportanceReport(importanceType: String,
                                   importanceDescription: String,
                                   featureImportance: Map[(String, String), (Int, Double, String)],
                                   rankToImportance: Map[Double, Double]) extends LogicalReport