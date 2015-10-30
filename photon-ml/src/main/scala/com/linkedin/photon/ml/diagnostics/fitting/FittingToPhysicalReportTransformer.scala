package com.linkedin.photon.ml.diagnostics.fitting

import com.linkedin.photon.ml.diagnostics.reporting._
import com.xeiam.xchart.{StyleManager, ChartBuilder}

class FittingToPhysicalReportTransformer extends LogicalToPhysicalReportTransformer[FittingReport, SectionPhysicalReport] {
  import FittingToPhysicalReportTransformer._

  def transform(report:FittingReport): SectionPhysicalReport = {
    val plotsSubsection = transformMetricsSection(report)
    val messagesSubsection = transformMessageSection(report)
    new SectionPhysicalReport(messagesSubsection :: plotsSubsection.toList, SECTION_NAME)
  }

  private def transformMessageSection(report:FittingReport):SectionPhysicalReport = {
    new SectionPhysicalReport(Seq(new SimpleTextPhysicalReport(report.fittingMsg)), MESSAGE_SECTION_NAME)
  }

  private def transformMetricsSection(report:FittingReport):Option[SectionPhysicalReport] = {
    if (report.metrics.isEmpty) {
      None
    } else {
      Some(new SectionPhysicalReport(report.metrics.keys.toSeq.sorted.map(x => {
        val (xData, trainData, testData) = report.metrics.get(x).get
        val minMetric = math.min(trainData.fold(trainData.last)((a, b) => math.min(a, b)), testData.fold(testData.last)((a, b) => math.min(a, b)))
        val maxMetric = math.max(trainData.fold(trainData.last)((a, b) => math.max(a, b)), testData.fold(testData.last)((a, b) => math.max(a, b)))
        val range = maxMetric - minMetric
        val yMin = if (math.abs(range) < 1e-15) { 0.9 * (minMetric + maxMetric) / 2.0 } else { minMetric - 0.05 * range }
        val yMax = if (math.abs(range) < 1e-15) { 1.1 * (minMetric + maxMetric) / 2.0 } else { maxMetric + 0.05 * range }

        if (minMetric.isInfinite || minMetric.isNaN || maxMetric.isInfinite || maxMetric.isNaN) {
          new SimpleTextPhysicalReport(
            s"""
               |Failed to get valid range for metric $x. Training set portion: [${xData.mkString(",")}].
               |Performance on training set: [${trainData.mkString(",")}.
               |Performance on holdout set: [${testData.mkString(",")}.
             """.stripMargin)
        } else {
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
          chart.getStyleManager.setXAxisMax(1.0)
          chart.getStyleManager.setYAxisMin(yMin)
          chart.getStyleManager.setYAxisMax(yMax)
          new PlotPhysicalReport(chart)
        }
      }), PLOTS_SECTION_NAME))
    }
  }
}

object FittingToPhysicalReportTransformer {
  def SECTION_NAME = "Fit Analysis"
  def PLOTS_SECTION_NAME = "Metric Plots"
  def MESSAGE_SECTION_NAME = "Messages"
}