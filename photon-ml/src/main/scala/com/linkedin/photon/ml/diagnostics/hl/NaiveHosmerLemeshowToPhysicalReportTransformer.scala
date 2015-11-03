package com.linkedin.photon.ml.diagnostics.hl

import com.linkedin.photon.ml.diagnostics.reporting._
import com.xeiam.xchart.{StyleManager, ChartBuilder, QuickChart, Chart}

/**
 * Simple, naive transformation (don't attempt to do anything clever with plots, for example) for HL test results
 */
class NaiveHosmerLemeshowToPhysicalReportTransformer extends LogicalToPhysicalReportTransformer[HosmerLemeshowReport, SectionPhysicalReport] {

  import NaiveHosmerLemeshowToPhysicalReportTransformer._

  def transform(hlr: HosmerLemeshowReport): SectionPhysicalReport = {
    val plot = generatePlots(hlr)
    val explanatoryText = generateExplanatoryText(hlr)
    new SectionPhysicalReport(Seq(plot, explanatoryText), SECTION_HEADER)
  }

  private def generatePlots(hlr: HosmerLemeshowReport): SectionPhysicalReport = {
    val plots = Seq(
      plotObservedExpected(hlr.histogram),
      plotCounts(hlr.histogram),
      plotCumulativeCounts(hlr.histogram),
      plotPositiveToNegativePortion(hlr.histogram))

    new SectionPhysicalReport(plots.map(x => new PlotPhysicalReport(x)), PLOT_HEADER)
  }

  private def generateExplanatoryText(hlr: HosmerLemeshowReport): SectionPhysicalReport = {
    val binningMsg = new SectionPhysicalReport(Seq(new SimpleTextPhysicalReport(hlr.binningMsg)), BINNING_HEADER)
    val chisquareCalcMsg = new SectionPhysicalReport(Seq(new SimpleTextPhysicalReport(hlr.chiSquareCalculationMsg)), CHI_SQUARE_HEADER)
    val cutoffMsg = new BulletedListPhysicalReport(hlr.getCutoffAnalysis.map(x => new SimpleTextPhysicalReport(x)))
    val analysisMsg = new BulletedListPhysicalReport(Seq(
      new SimpleTextPhysicalReport(hlr.getTestDescription()),
      new SimpleTextPhysicalReport(hlr.getPointProbabilityAnalysis()),
      cutoffMsg))
    new SectionPhysicalReport(Seq(analysisMsg, binningMsg, chisquareCalcMsg), ANALYSIS_HEADER)
  }
}

object NaiveHosmerLemeshowToPhysicalReportTransformer {
  val SECTION_HEADER = "Hosmer-Lemeshow Goodness-of-Fit Test for Logistic Regression"
  val ANALYSIS_HEADER = "Analysis"
  val PLOT_HEADER = "Plots"
  val BINNING_HEADER = "Messages generated during histogram calculation"
  val CHI_SQUARE_HEADER = "Messages generated during Chi square calculation"
  val LABEL_BREAKDOWN_TITLE = "Count by Score"
  val CUMULATIVE_LABEL_BREAKDOWN_TITLE = "Cumulative count by Score"

  private def plotObservedExpected(x:Array[PredictedProbabilityVersusObservedFrequencyHistogramBin]): Chart = {
    val xSeries = x.map(b => 100.0 * (b.lowerBound + b.upperBound) / 2.0)
    val ySeries1 = x.map(b => {
      val totalCount = b.observedNegCount + b.observedPosCount
      val percent = if (totalCount > 0) b.observedPosCount.toDouble / totalCount else 0.0
      100.0 * percent
    })
    val ySeries2 = xSeries
    val builder = new ChartBuilder()
    builder
      .chartType(StyleManager.ChartType.Bar)
      .theme(StyleManager.ChartTheme.XChart)
      .title("Observed positive rate versus predicted positive rate")
      .xAxisTitle("Predicted positive rate")
      .yAxisTitle("Observed positive rate")
      .width(PlotUtils.PLOT_WIDTH)
      .height(PlotUtils.PLOT_HEIGHT)
    val chart = builder.build()
    chart.addSeries("Observed", xSeries, ySeries1)
    chart.addSeries("Expected", xSeries, ySeries2)
    chart.getStyleManager.setXAxisMin(0.0)
    chart.getStyleManager.setXAxisMax(100.0)
    chart.getStyleManager.setYAxisMin(0.0)
    chart.getStyleManager.setYAxisMax(100.0)
    chart
  }


  private def plotPositiveToNegativePortion(x:Array[PredictedProbabilityVersusObservedFrequencyHistogramBin]): Chart = {
    val (posCount, negCount) = x.foldLeft((0L, 0L))((prev, bin) => {
      (prev._1 + bin.observedPosCount, prev._2 + bin.observedNegCount)
    })

    val builder = new ChartBuilder()

    builder
      .chartType(StyleManager.ChartType.Bar)
      .theme(StyleManager.ChartTheme.XChart)
      .title(LABEL_BREAKDOWN_TITLE)
      .xAxisTitle("")
      .yAxisTitle("Count")
      .width(PlotUtils.PLOT_WIDTH)
      .height(PlotUtils.PLOT_HEIGHT)
    val chart = builder.build()
    chart.addSeries("Positive", Array(0.0), Array(posCount.toDouble))
    chart.addSeries("Negative", Array(0.0), Array(negCount.toDouble))
    chart
  }

  private def plotCounts(x:Array[PredictedProbabilityVersusObservedFrequencyHistogramBin]): Chart = {
    val xSeries = x.map(y => (y.lowerBound + y.upperBound)/2.0)
    val posSamples = x.map(_.observedPosCount.toDouble)
    val negSamples = x.map(_.observedNegCount.toDouble)
    val totalSamples = x.map(y => (y.observedNegCount + y.observedPosCount).toDouble)


    val builder = new ChartBuilder()
    builder
      .chartType(StyleManager.ChartType.Bar)
      .theme(StyleManager.ChartTheme.XChart)
      .title(LABEL_BREAKDOWN_TITLE)
      .xAxisTitle("Score")
      .yAxisTitle("Count")
      .width(PlotUtils.PLOT_WIDTH)
      .height(PlotUtils.PLOT_HEIGHT)
    val chart = builder.build()

    chart.getStyleManager.setXAxisMin(0.0)
    chart.getStyleManager.setXAxisMax(100.0)
    chart.addSeries("Positive", xSeries, posSamples)
    chart.addSeries("Negative", xSeries, negSamples)
    chart.addSeries("Total", xSeries, totalSamples)

    chart
  }

  private def plotCumulativeCounts(x:Array[PredictedProbabilityVersusObservedFrequencyHistogramBin]): Chart = {
    val xSeries = x.map(y => (y.lowerBound + y.upperBound)/2.0)
    val posSamples = x.map(_.observedPosCount.toDouble).scanLeft(0.0)(_+_).tail
    val negSamples = x.map(_.observedNegCount.toDouble).scanLeft(0.0)(_+_).tail
    val totalSamples = x.map(y => (y.observedNegCount + y.observedPosCount).toDouble).scanLeft(0.0)(_+_).tail


    val builder = new ChartBuilder()
    builder
      .chartType(StyleManager.ChartType.Bar)
      .theme(StyleManager.ChartTheme.XChart)
      .title(CUMULATIVE_LABEL_BREAKDOWN_TITLE)
      .xAxisTitle("Score")
      .yAxisTitle("Cumulative Count")
      .width(PlotUtils.PLOT_WIDTH)
      .height(PlotUtils.PLOT_HEIGHT)
    val chart = builder.build()

    chart.getStyleManager.setXAxisMin(0.0)
    chart.getStyleManager.setXAxisMax(100.0)
    chart.addSeries("Positive", xSeries, posSamples)
    chart.addSeries("Negative", xSeries, negSamples)
    chart.addSeries("Total", xSeries, totalSamples)

    chart
  }
}
