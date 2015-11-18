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

import com.linkedin.photon.ml.diagnostics.reporting._
import com.xeiam.xchart.{StyleManager, ChartBuilder}

/**
 * Prepare a feature importance report for presentation
 */
class FeatureImportanceToPhysicalReportTransformer extends LogicalToPhysicalReportTransformer[FeatureImportanceReport, SectionPhysicalReport] {

  import FeatureImportanceToPhysicalReportTransformer._

  def transform(report: FeatureImportanceReport): SectionPhysicalReport = {
    val plot = generatePlot(report)
    val description = generateDescription(report)
    new SectionPhysicalReport(Seq(plot, description), s"$SECTION_TITLE_PREFIX [${report.importanceType}]")
  }

  def generatePlot(report: FeatureImportanceReport): PlotPhysicalReport = {
    val builder = new ChartBuilder()
    val chart = builder
      .chartType(StyleManager.ChartType.Area)
      .height(PlotUtils.PLOT_HEIGHT)
      .theme(StyleManager.ChartTheme.XChart)
      .title(report.importanceType)
      .width(PlotUtils.PLOT_WIDTH)
      .xAxisTitle("% features with greater importance")
      .yAxisTitle("Relative importance")
      .build()
    val sorted = report.rankToImportance.toList.sortBy(_._1)
    val xSeries = sorted.map(_._1).toArray
    val ySeries = sorted.map(_._2).toArray
    chart.addSeries(report.importanceDescription, xSeries, ySeries)
    chart.getStyleManager.setXAxisMin(0)
    chart.getStyleManager.setXAxisMax(100)
    new PlotPhysicalReport(chart, None)
  }

  def generateDescription(report: FeatureImportanceReport): BulletedListPhysicalReport = {
    new BulletedListPhysicalReport(report.featureImportance.toList.sortBy(_._2._2).reverse.map(x => new SimpleTextPhysicalReport(x._2._3)))
  }
}

object FeatureImportanceToPhysicalReportTransformer {
  val SECTION_TITLE_PREFIX = "Feature importance"
  val PLOT_SECTION_TITLE = "Importance Distribution"
  val DESCRIPTION_SECTION_TITLE = "Analysis"
}
