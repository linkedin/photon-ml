package com.linkedin.photon.ml.diagnostics.reporting.reports.model

import com.linkedin.photon.ml.diagnostics.bootstrap.BootstrapToPhysicalReportTransformer
import com.linkedin.photon.ml.diagnostics.featureimportance.FeatureImportanceToPhysicalReportTransformer
import com.linkedin.photon.ml.diagnostics.fitting.FittingToPhysicalReportTransformer
import com.linkedin.photon.ml.diagnostics.hl.NaiveHosmerLemeshowToPhysicalReportTransformer
import com.linkedin.photon.ml.diagnostics.independence.PredictionErrorIndependencePhysicalReportTransformer
import com.linkedin.photon.ml.diagnostics.reporting._
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel

/**
 * Convert model diagnostics into a presentable form.
 */
class ModelDiagnosticToPhysicalReportTransformer[GLM <: GeneralizedLinearModel]
  extends LogicalToPhysicalReportTransformer[ModelDiagnosticReport[GLM], SectionPhysicalReport] {

  import ModelDiagnosticToPhysicalReportTransformer._

  def transform(model: ModelDiagnosticReport[GLM]): SectionPhysicalReport = {
    val metricsSection:SectionPhysicalReport = transformMetrics(model)
    val predErrSection:SectionPhysicalReport = PREDICTION_ERROR_TRANSFORMER
      .transform(model.predictionErrorIndependence)
    val modelSection: SectionPhysicalReport = new SectionPhysicalReport(
      Seq(
        FEATURE_IMPORTANCE_TRANSFORMER.transform(model.meanImpactFeatureImportance),
        FEATURE_IMPORTANCE_TRANSFORMER.transform(model.varianceImpactFeatureImportance)),
      FEATURE_IMPORTANCE_TITLE)

    val hlSection = model.hosmerLemeshow match {
      case Some(hl) =>
        Some(HOSMER_LEMESHOW_TRANSFORMER.transform(hl))
      case None =>
        None
    }

    val fitSection = model.fitReport match {
      case Some(fr) =>
        Some(FIT_TRANSFORMER.transform(fr))
      case None =>
        println("Did not get a matching model fit report!")
        None
    }

    val bootstrapSection = model.bootstrapReport match {
      case Some(b) => Some(BOOTSTRAP_TRANSFORMER.transform(b))
      case None => None
    }

    new SectionPhysicalReport(
      metricsSection :: predErrSection :: modelSection ::
      fitSection.toList ++ bootstrapSection.toList ++ hlSection.toList,
      s"$SECTION_TITLE: ${model.modelDescription}, lambda=${model.lambda}")
  }

  private def transformMetrics(model:ModelDiagnosticReport[GLM]): SectionPhysicalReport = {
    new SectionPhysicalReport(
      Seq(
        new BulletedListPhysicalReport(
          model.metrics
            .map(x => s"Metric: [${x._1}, value: [${x._2}]")
            .toSeq
            .sorted
            .map(x => new SimpleTextPhysicalReport(x)))),
      "Validation Set Metrics")
  }
}

object ModelDiagnosticToPhysicalReportTransformer {
  val SECTION_TITLE = "Model Analysis"
  val HOSMER_LEMESHOW_TRANSFORMER = new NaiveHosmerLemeshowToPhysicalReportTransformer()
  val FEATURE_IMPORTANCE_TRANSFORMER = new FeatureImportanceToPhysicalReportTransformer()
  val PREDICTION_ERROR_TRANSFORMER = new PredictionErrorIndependencePhysicalReportTransformer()
  val FIT_TRANSFORMER = new FittingToPhysicalReportTransformer()
  val BOOTSTRAP_TRANSFORMER = new BootstrapToPhysicalReportTransformer()
  val FEATURE_IMPORTANCE_TITLE = "Coefficient Importance Analysis"
}
