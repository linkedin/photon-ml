/*
 * Copyright 2017 LinkedIn Corp. All rights reserved.
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
package com.linkedin.photon.ml.diagnostics.independence

import com.linkedin.photon.ml.diagnostics.reporting._
import com.xeiam.xchart.{StyleManager, ChartBuilder}

class PredictionErrorIndependencePhysicalReportTransformer
  extends LogicalToPhysicalReportTransformer[PredictionErrorIndependenceReport, SectionPhysicalReport] {

  import PredictionErrorIndependencePhysicalReportTransformer._

  override def transform(report: PredictionErrorIndependenceReport): SectionPhysicalReport = {
    val plotSection = generatePlot(report.errorSample, report.predictionSample)
    val kendallSection = generateKendall(report.kendallTau)
    new SectionPhysicalReport(Seq(plotSection, kendallSection), SECTION_TITLE)
  }
}

object PredictionErrorIndependencePhysicalReportTransformer {
  val SECTION_TITLE = "Error / Prediction Independence Analysis"
  val PARAMETRIC_TEST_SUBSECTION = "Parametric Tests of Independence"
  val PLOT_SUBSECTION_TITLE = "Plot"
  val CHI_SQUARE_SUBSECTION_TITLE = "Chi-square Independence Test"
  val G_SUBSECTION_TITLE = "G Independence Test"
  val KENDALL_TAU_SECTION_TITLE = "Kendall Tau Independence Test"
  val CONTINGENCY_TABLE_SECTION_TITLE = "Contingency Table"

  def generatePlot(error:Array[Double], prediction:Array[Double]): SectionPhysicalReport = {
    val builder = new ChartBuilder
    val chart = builder.chartType(StyleManager.ChartType.Scatter)
                       .height(PlotUtils.PLOT_HEIGHT)
                       .theme(StyleManager.ChartTheme.XChart)
                       .title("Error v. Prediction")
                       .width(PlotUtils.PLOT_WIDTH)
                       .xAxisTitle("Prediction")
                       .yAxisTitle("Label - Prediction")
                       .build()

    val xRange = PlotUtils.getRange(prediction)
    val yRange = PlotUtils.getRange(error)

    chart.addSeries("Prediction error", prediction, error)
    chart.getStyleManager.setXAxisMin(xRange._1)
    chart.getStyleManager.setXAxisMax(xRange._2)
    chart.getStyleManager.setYAxisMin(yRange._1)
    chart.getStyleManager.setYAxisMax(yRange._2)

    new SectionPhysicalReport(Seq(new PlotPhysicalReport(chart)), PLOT_SUBSECTION_TITLE)
  }

  def generateKendall(report:KendallTauReport): SectionPhysicalReport = {
    new SectionPhysicalReport(
      Seq(new BulletedListPhysicalReport(
        Seq(
          new SimpleTextPhysicalReport(s"Concordant pairs: ${report.concordantPairs}"),
          new SimpleTextPhysicalReport(s"Discordant pairs: ${report.discordantPairs}"),
          new SimpleTextPhysicalReport(s"Effective pairs: ${report.effectivePairs}"),
          new SimpleTextPhysicalReport(s"Number of samples: ${report.numSamples}"),
          new SimpleTextPhysicalReport(s"Tau alpha: ${report.tauAlpha}"),
          new SimpleTextPhysicalReport(s"Tau beta: ${report.tauBeta}"),
          new SimpleTextPhysicalReport(s"Z alpha: ${report.zAlpha}"),
          new SimpleTextPhysicalReport(s"Alpha p-value: ${report.pValueAlpha}")
        )
      )), KENDALL_TAU_SECTION_TITLE)
  }
}
