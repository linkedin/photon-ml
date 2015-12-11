package com.linkedin.photon.ml.diagnostics.fitting

import com.linkedin.photon.ml.Evaluation
import com.linkedin.photon.ml.diagnostics.reporting._
import com.xeiam.xchart.{StyleManager, ChartBuilder}

class FittingToPhysicalReportTransformer
  extends LogicalToPhysicalReportTransformer[FittingReport, SectionPhysicalReport] {

  import FittingToPhysicalReportTransformer._

  def transform(report: FittingReport): SectionPhysicalReport = {
    val plotsSubsection = transformMetricsSection(report)
    val messagesSubsection = transformMessageSection(report)
    new SectionPhysicalReport(messagesSubsection :: plotsSubsection.toList, SECTION_NAME)
  }

  private def transformMessageSection(report: FittingReport): SectionPhysicalReport = {
    new SectionPhysicalReport(Seq(new SimpleTextPhysicalReport(report.fittingMsg)), MESSAGE_SECTION_NAME)
  }

  private def transformMetricsSection(report: FittingReport): Option[SectionPhysicalReport] = {
    if (report.metrics.isEmpty) {
      None
    } else {
      Some(new SectionPhysicalReport(report.metrics.keys.toSeq.sorted.map(x => {
        val (xData, trainData, testData) = report.metrics.get(x).get
        val range = PlotUtils.getRangeForMetric(x, trainData ++ testData)

        val builder = new ChartBuilder()
        val chart = builder.chartType(StyleManager.ChartType.Line)
          .height(PlotUtils.PLOT_HEIGHT)
          .theme(StyleManager.ChartTheme.XChart)
          .title(x)
          .width(PlotUtils.PLOT_WIDTH)
          .xAxisTitle("Portion of training set")
          .yAxisTitle("Metric value")
          .build()
        chart.addSeries("Training set", xData, trainData)
        chart.addSeries("Holdout set", xData, testData)
        chart.getStyleManager.setXAxisMin(0.0)
        chart.getStyleManager.setXAxisMax(100.0)
        chart.getStyleManager.setYAxisMin(range._1)
        chart.getStyleManager.setYAxisMax(range._2)
        new PlotPhysicalReport(chart)
      }), PLOTS_SECTION_NAME))
    }
  }
}

object FittingToPhysicalReportTransformer {
  def SECTION_NAME = "Fit Analysis"

  def PLOTS_SECTION_NAME = "Metric Plots"

  def MESSAGE_SECTION_NAME = "Messages"
}
