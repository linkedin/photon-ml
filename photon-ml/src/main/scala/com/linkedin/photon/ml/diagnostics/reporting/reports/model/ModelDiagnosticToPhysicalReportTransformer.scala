package com.linkedin.photon.ml.diagnostics.reporting.reports.model

import com.linkedin.photon.ml.diagnostics.hl.NaiveHosmerLemeshowToPhysicalReportTransformer
import com.linkedin.photon.ml.diagnostics.reporting._
import com.linkedin.photon.ml.diagnostics.reporting.reports.Utils
import com.linkedin.photon.ml.stat.BasicStatisticalSummary
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import com.xeiam.xchart.{StyleManager, ChartBuilder}

/**
 * Convert model diagnostics into a presentable form.
 */
class ModelDiagnosticToPhysicalReportTransformer[GLM <: GeneralizedLinearModel] extends LogicalToPhysicalReportTransformer[ModelDiagnosticReport[GLM], SectionPhysicalReport] {

  import ModelDiagnosticToPhysicalReportTransformer._

  def transform(model: ModelDiagnosticReport[GLM]): SectionPhysicalReport = {
    val modelSection: SectionPhysicalReport = transformModel(model.model, model.modelDescription, model.nameIdxMap, model.summary)
    model.hosmerLemeshow match {
      case Some(hl) =>
        val hlSection: SectionPhysicalReport = HOSMER_LEMESHOW_TRANSFORMER.transform(hl)
        new SectionPhysicalReport(Seq(modelSection, hlSection), f"Model, lambda=${model.lambda}%.03g")
      case None =>
        new SectionPhysicalReport(Seq(modelSection), f"Model, lambda=${model.lambda}%.03g")
    }
  }

  private def transformModel(m: GLM, desc: String, nameIdx: Map[String, Int], summary: Option[BasicStatisticalSummary]): SectionPhysicalReport = {
    summary match {
      case Some(sum) =>
        transformModel(m, desc, nameIdx, sum)
      case None =>
        transformModel(m, desc, nameIdx)
    }
  }

  private def transformModel(m: GLM, desc: String, nameIdx: Map[String, Int], summary: BasicStatisticalSummary): SectionPhysicalReport = {
    val coefficientImportance = (for {coeff <- m.coefficients.valuesIterator.zipWithIndex} yield (coeff._2, math.abs(coeff._1 * summary.meanAbs(coeff._2)), coeff._1, summary.mean(coeff._2))).toArray.sortBy(x => x._1)

    val coefficientImportanceMap = nameIdx.map(x => {
      (x._1, coefficientImportance(x._2))
    })

    val sortedImportance = coefficientImportanceMap.toArray.sortBy(x => x._2._2)

    renderImportance("Magnitude of expected inner product contribution", sortedImportance)
  }

  private def transformModel(m: GLM, desc: String, nameIdx: Map[String, Int]): SectionPhysicalReport = {
    val coefficientImportance = (for {coeff <- m.coefficients.valuesIterator.zipWithIndex} yield (coeff._2, math.abs(coeff._1), coeff._1, 1.0)).toArray.sortBy(x => x._1)

    val coefficientImportanceMap = nameIdx.map(x => {
      (x._1, coefficientImportance(x._2))
    })

    val sortedImportance = coefficientImportanceMap.toArray.sortBy(x => x._2._2)

    renderImportance("Magnitude of coefficient", sortedImportance)
  }

  private def renderImportance(coeffImportDesc: String, sortedImportance: Array[(String, (Int, Double, Double, Double))]): SectionPhysicalReport = {
    // Thing 1: draw importance at 1% increments
    val featImportanceIdx = (1 until 99).map(x => x * sortedImportance.length / 100)
    val featImportanceY = featImportanceIdx.map(x => sortedImportance(x)._2._2).toArray
    val featImportanceX = featImportanceIdx.map(x => 100.0 * x / sortedImportance.length).toArray
    val builder = new ChartBuilder()
    val chart = builder
      .chartType(StyleManager.ChartType.Line)
      .height(PLOT_HEIGHT)
      .width(PLOT_WIDTH)
      .theme(StyleManager.ChartTheme.XChart)
      .title(MODEL_IMPORTANCE_TITLE)
      .xAxisTitle("Rank (importance %-ile)")
      .yAxisTitle("Importance")
      .build()
    chart.addSeries(coeffImportDesc, featImportanceX, featImportanceY)
    val plot = new PlotPhysicalReport(chart)

    // Thing 2: pull out description of most important features
    val importantFeatures = new NumberedListPhysicalReport(sortedImportance.takeRight(MAX_IMPORTANT_FEATURES).reverse.map(x => {
      val (name, term) = Utils.extractNameTerm(x._1)
      val (coeffIdx, coeffImp, coeffVal, expFeat) = x._2
      new SimpleTextPhysicalReport(f"Feature (N: [$name] T:[$term]) has importance $coeffImp%.04g (coefficient value: $coeffVal, expected feature value: $expFeat)")
    }))

    new SectionPhysicalReport(Seq(plot, importantFeatures), FEATURE_IMPORTANCE_TITLE)
  }
}

object ModelDiagnosticToPhysicalReportTransformer {
  val HOSMER_LEMESHOW_TRANSFORMER = new NaiveHosmerLemeshowToPhysicalReportTransformer()
  val MAX_IMPORTANT_FEATURES = 30
  val MODEL_IMPORTANCE_TITLE = "Model Coefficient Importance"
  val FEATURE_IMPORTANCE_TITLE = "Coefficient Importance Analysis"
  val PLOT_HEIGHT = 960
  val PLOT_WIDTH = 1280
}
