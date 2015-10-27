package com.linkedin.photon.ml.diagnostics.reporting.reports.model

import com.linkedin.photon.ml.diagnostics.featureimportance.FeatureImportanceToPhysicalReportTransformer
import com.linkedin.photon.ml.diagnostics.hl.NaiveHosmerLemeshowToPhysicalReportTransformer
import com.linkedin.photon.ml.diagnostics.reporting._
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel

/**
 * Convert model diagnostics into a presentable form.
 */
class ModelDiagnosticToPhysicalReportTransformer[GLM <: GeneralizedLinearModel] extends LogicalToPhysicalReportTransformer[ModelDiagnosticReport[GLM], SectionPhysicalReport] {

  import ModelDiagnosticToPhysicalReportTransformer._

  def transform(model: ModelDiagnosticReport[GLM]): SectionPhysicalReport = {
    val metricsSection:SectionPhysicalReport = transformMetrics(model)
    val modelSection: SectionPhysicalReport = new SectionPhysicalReport(
      Seq(
        FEATURE_IMPORTANCE_TRANSFORMER.transform(model.meanImpactFeatureImportance),
        FEATURE_IMPORTANCE_TRANSFORMER.transform(model.varianceImpactFeatureImportance)),
      FEATURE_IMPORTANCE_TITLE)

    model.hosmerLemeshow match {
      case Some(hl) =>
        val hlSection: SectionPhysicalReport = HOSMER_LEMESHOW_TRANSFORMER.transform(hl)
        new SectionPhysicalReport(Seq(metricsSection, modelSection, hlSection), f"Model, lambda=${model.lambda}%.03g")
      case None =>
        new SectionPhysicalReport(Seq(metricsSection, modelSection), f"Model, lambda=${model.lambda}%.03g")
    }
  }

  private def transformMetrics(model:ModelDiagnosticReport[GLM]): SectionPhysicalReport = {
    new SectionPhysicalReport(
      Seq(
        new BulletedListPhysicalReport(model.metrics.map(x => s"Metric: [${x._1}, value: [${x._2}]").toSeq.sorted.map(x => new SimpleTextPhysicalReport(x)))),
      "Validation Set Metrics")
  }


}

object ModelDiagnosticToPhysicalReportTransformer {
  val HOSMER_LEMESHOW_TRANSFORMER = new NaiveHosmerLemeshowToPhysicalReportTransformer()
  val FEATURE_IMPORTANCE_TRANSFORMER = new FeatureImportanceToPhysicalReportTransformer()
  val MAX_IMPORTANT_FEATURES = 30
  val MODEL_IMPORTANCE_TITLE = "Model Coefficient Importance"
  val FEATURE_IMPORTANCE_TITLE = "Coefficient Importance Analysis"
}
