package com.linkedin.photon.ml.diagnostics.featureimportance

import com.linkedin.photon.ml.diagnostics.reporting.reports.Utils
import com.linkedin.photon.ml.stat.BasicStatisticalSummary
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel


/**
 * Feature importance defined by impact on inner product <em>expectation</em> in contrast to [[VarianceFeatureImportanceDiagnostic]]
 * which looks at impact on inner product <em>variance</em>
 *
 * @param modelNameToIndex
 * Map of encoded name/term &rarr; feature index
 */

class ExpectedMagnitudeFeatureImportanceDiagnostic(modelNameToIndex: Map[String, Int]) extends AbstractFeatureImportanceDiagnostic(modelNameToIndex) {
  protected def getImportanceType(): String = {
    "Inner product expectation"
  }

  protected def getImportanceDescription(summary: Option[BasicStatisticalSummary]): String = {
    summary match {
      case Some(_) => "Expected magnitude of inner product contribution"
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
          sum.meanAbs(x._2)
        case None => 1.0
      }
      val importance = math.abs(coeff * expAbs)
      (nameTerm, index, importance)
    })
  }
}
