/*
 * Copyright 2015 LinkedIn Corp. All rights reserved.
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

import com.linkedin.photon.ml.diagnostics.reporting.reports.Utils
import com.linkedin.photon.ml.stat.BasicStatisticalSummary
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel


/**
 * Feature importance defined by impact on inner product <em>variance</em> in contrast to [[ExpectedMagnitudeFeatureImportanceDiagnostic]]
 * which looks at impact on inner product <em>expectation</em>
 *
 * @param modelNameToIndex
 * Map of encoded name/term &rarr; feature index
 */
class VarianceFeatureImportanceDiagnostic(modelNameToIndex: Map[String, Int]) extends AbstractFeatureImportanceDiagnostic(modelNameToIndex) {
  protected def getImportanceType(): String = {
    "Inner product variance"
  }

  protected def getImportanceDescription(summary: Option[BasicStatisticalSummary]): String = {
    summary match {
      case Some(_) => "Expected inner product variance contribution"
      case None => "Magnitude of feature coefficient"
    }
  }

  protected def getImportances(model: GeneralizedLinearModel, summary: Option[BasicStatisticalSummary]): Iterable[((String, String), Int, Double)] = {
    modelNameToIndex.map(x => {
      val nameTerm = Utils.extractNameTerm(x._1)
      val index = x._2
      val coeff = model.coefficients(index)
      val expAbs = summary match {
        case Some(sum) =>
          sum.variance(x._2)
        case None => 1.0
      }
      val importance = math.abs(coeff * expAbs)
      (nameTerm, index, importance)
    })
  }
}
