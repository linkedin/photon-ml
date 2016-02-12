package com.linkedin.photon.ml.diagnostics.reporting.reports.combined

import java.text.SimpleDateFormat
import java.util.{Date, TimeZone, Calendar}

import com.linkedin.photon.ml.Evaluation
import com.linkedin.photon.ml.diagnostics.reporting._
import com.linkedin.photon.ml.diagnostics.reporting.reports.model.{
  ModelDiagnosticReport, ModelDiagnosticToPhysicalReportTransformer}
import com.linkedin.photon.ml.diagnostics.reporting.reports.system.SystemToPhysicalReportTransformer
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import com.xeiam.xchart.{StyleManager, ChartBuilder}

/**
 * Transform diagnostic reports into their physical report representation
 */
class DiagnosticToPhysicalReportTransformer
  extends LogicalToPhysicalReportTransformer[DiagnosticReport, DocumentPhysicalReport] {

  import DiagnosticToPhysicalReportTransformer._

  def transform(diag: DiagnosticReport): DocumentPhysicalReport = {
    val formatter = new SimpleDateFormat()
    val now = new Date()
    val lambdaReports = diag.modelReports
      .sortBy(_.lambda)
      .map(x => (x.lambda, (x, MODEL_SECTION_TRANSFORMER.transform(x)))).toMap
    val summary = transformSummary(lambdaReports)

    new DocumentPhysicalReport(
      Seq(
        new ChapterPhysicalReport(Seq(summary), "Summary"),
        SYSTEM_CHAPTER_TRANSFORMER.transform(diag.systemReport),
        new ChapterPhysicalReport
        (lambdaReports.toSeq.sortBy(_._1).map(_._2._2) , MODEL_CHAPTER_TITLE)),
      s"Modeling run ${formatter.format(now)}")
  }
}

object DiagnosticToPhysicalReportTransformer {
  val SYSTEM_CHAPTER_TRANSFORMER = new SystemToPhysicalReportTransformer()
  val MODEL_SECTION_TRANSFORMER = new ModelDiagnosticToPhysicalReportTransformer[GeneralizedLinearModel]()
  val MODEL_CHAPTER_TITLE = "Detailed Model Diagnostics"
  val MODEL_SUMMARY_CHAPTER = "Summary"
  val MODEL_METRICS_SUMMARY = "Model Metrics"

  private def getBestModelByMetric(metricsByLambda: Map[String, Map[Double, Double]]) = {
    val metrics = metricsByLambda.keys.toSeq.sorted

    new BulletedListPhysicalReport(
      metrics.flatMap(metric => {
        val metadata = Evaluation.metricMetadata.get(metric).get

        metricsByLambda.get(metric).toSeq.flatMap(values => {
          val ordering = new Ordering[(Double, Double)]() {
            override def compare(x: (Double, Double), y: (Double, Double)): Int =
              metadata.worstToBestOrdering.compare(x._2, y._2)
          }
          val sorted = values.toSeq.sorted(ordering)
          val bestLambda = sorted.last

          Seq(new SimpleTextPhysicalReport(s"Metric ${metric} best: ${bestLambda._2} @ lambda = ${bestLambda._1}"))
            .iterator
        }).iterator
      }))
  }

  private def getModelMetricPlots(metricsByLambda: Map[String, Map[Double, Double]]) = {
    val metrics = metricsByLambda.keys.toSeq.sorted

    metrics.flatMap(metric => {
      val metadata = Evaluation.metricMetadata.get(metric).get

      metricsByLambda.get(metric).toSeq.flatMap(lambdaToMetric => {
        val sortedByLambda = lambdaToMetric.toSeq.sortBy(_._1)
        val builder = new ChartBuilder()
        val chart = builder.chartType(StyleManager.ChartType.Bar)
          .height(PlotUtils.PLOT_HEIGHT)
          .theme(StyleManager.ChartTheme.XChart)
          .title(metric)
          .width(PlotUtils.PLOT_WIDTH)
          .build()

        sortedByLambda.foreach(x => {
          chart.addSeries(s"Lambda = ${x._1}", Array(1.0), Array(x._2))
        })

        val yRange = PlotUtils.getRangeForMetric(metric, sortedByLambda.map(_._2))
        chart.getStyleManager.setYAxisMin(yRange._1)
        chart.getStyleManager.setYAxisMax(yRange._2)

        Seq(new PlotPhysicalReport(chart)).iterator
      }).iterator
    })
  }

  /**
   * Generate a summary section which includes a quick description of which lambdas did best/worst on a particular
   * metric and a visual summary of models by metric.
   *
   * @param reports Map of &lambda; &rarr; model diagnostic for that lambda
   * @return
   */
  private def transformSummary(
      reports: Map[Double, (ModelDiagnosticReport[GeneralizedLinearModel], SectionPhysicalReport)]):
        SectionPhysicalReport = {

    val metricsByLambda = reports.map(_._2._1).flatMap(x => {
      x.metrics.map(y => {
        (y._1, (x.lambda, y._2))
      }).iterator
    }).groupBy(_._1).mapValues(_.map(_._2).toMap)

    val bestModelByMetric = getBestModelByMetric(metricsByLambda)
    val modelMetricPlots = getModelMetricPlots(metricsByLambda)

    val links = new BulletedListPhysicalReport(reports.toSeq.sortBy(_._1).map(x => {
      new ReferencePhysicalReport(x._2._2, s"Jump to model with lambda = ${x._1}")
    }))

    new SectionPhysicalReport(Seq(links, bestModelByMetric) ++ modelMetricPlots, MODEL_SUMMARY_CHAPTER)
  }
}
