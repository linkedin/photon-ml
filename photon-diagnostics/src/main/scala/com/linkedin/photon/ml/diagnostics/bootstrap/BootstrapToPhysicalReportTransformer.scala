/*
 * Copyright 2016 LinkedIn Corp. All rights reserved.
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
package com.linkedin.photon.ml.diagnostics.bootstrap

import com.xeiam.xchart.{StyleManager, ChartBuilder}

import com.linkedin.photon.ml.diagnostics.reporting._

class BootstrapToPhysicalReportTransformer
  extends LogicalToPhysicalReportTransformer[BootstrapReport, SectionPhysicalReport] {

  import BootstrapToPhysicalReportTransformer._

  private def getBootstrapMetrics(logical: BootstrapReport): SectionPhysicalReport = {
    new SectionPhysicalReport(Seq(new BulletedListPhysicalReport(
      logical.bootstrappedModelMetrics.map(item => {
        new SimpleTextPhysicalReport(s"Metric: ${item._1}, value: ${item._2}")
      }).toSeq)), BAGGED_MODEL_METRICS_SECTION_TITLE)
  }

  private def getMetricsDistribution(logical: BootstrapReport): SectionPhysicalReport = {
    new SectionPhysicalReport(logical.metricDistributions.map(x => {
      val builder = new ChartBuilder
      val plot = builder.chartType(StyleManager.ChartType.Bar)
        .height(PlotUtils.PLOT_HEIGHT)
        .theme(StyleManager.ChartTheme.XChart)
        .title(s"Bootstrap distribution of ${x._1}")
        .width(PlotUtils.PLOT_WIDTH)
        .yAxisTitle(s"${x._1}")
        .build()

      plot.addSeries(s"Min: ${x._2._1}", Array(0.0), Array(x._2._1))
      plot.addSeries(s"Q1: ${x._2._2}", Array(0.0), Array(x._2._2))
      plot.addSeries(s"Median: ${x._2._3}", Array(0.0), Array(x._2._3))
      plot.addSeries(s"Q3: ${x._2._4}", Array(0.0), Array(x._2._4))
      plot.addSeries(s"Max: ${x._2._5}", Array(0.0), Array(x._2._5))
      plot.getStyleManager.setXAxisTicksVisible(false)
      plot.getStyleManager.setXAxisMin(0.9)
      plot.getStyleManager.setXAxisMax(1.1)

      val range = PlotUtils.getRangeForMetric(x._1, Seq(x._2._1, x._2._5))

      plot.getStyleManager.setYAxisMin(range._1)
      plot.getStyleManager.setYAxisMax(range._2)

      new PlotPhysicalReport(plot)
    }).toSeq, METRICS_DISTRIBUTION_SECTION_TITLE)
  }

  private def getImportantFeatures(logical: BootstrapReport): SectionPhysicalReport = {
    new SectionPhysicalReport(
      logical.importantFeatureCoefficientDistributions.map(x => {
        val ((name, term), summary) = x

        val builder = new ChartBuilder
        val plot = builder.chartType(StyleManager.ChartType.Bar)
          .height(PlotUtils.PLOT_HEIGHT)
          .theme(StyleManager.ChartTheme.XChart)
          .title(
            s"Coefficient distribution for N=$name, T=$term (mean = ${summary.getMean}, " +
            s"st.dev = ${summary.getStdDev})")
          .yAxisTitle("Coefficient value")
          .width(PlotUtils.PLOT_WIDTH)
          .build()

        val yRange = PlotUtils.getRange(Seq(summary.getMin, summary.getMax))

        plot.addSeries(s"Minimum ${summary.getMin}", Array(0.0), Array(summary.getMin))
        plot.addSeries(
          s"First quartile ${summary.estimateFirstQuartile()}", Array(0.0), Array(summary.estimateFirstQuartile()))
        plot.addSeries(s"Median ${summary.estimateMedian()}", Array(0.0), Array(summary.estimateMedian()))
        plot.addSeries(
          s"Third quartile ${summary.estimateThirdQuartile()}", Array(0.0), Array(summary.estimateThirdQuartile()))
        plot.addSeries(s"Maximum ${summary.getMax}", Array(0.0), Array(summary.getMax))
        plot.getStyleManager.setYAxisMin(yRange._1)
        plot.getStyleManager.setYAxisMax(yRange._2)
        new PlotPhysicalReport(plot)
      }).toSeq, IMPORTANT_FEATURES_SECTION_TITLE)
  }

  private def getStraddlingZeroSection(logical: BootstrapReport): SectionPhysicalReport = {
    new SectionPhysicalReport(
      Seq(
        new SimpleTextPhysicalReport(
          s"Total features with interquartile range straddling zero: ${logical.zeroCrossingFeatures.size}"),
        new BulletedListPhysicalReport(logical.zeroCrossingFeatures.toSeq.sortBy(_._2._2).reverseMap(x => {
          val ((name, term), (_, importance, coeff)) = x
          new SimpleTextPhysicalReport(s"Feature N=$name, T=$term with importance $importance ==> $coeff")
        }))), FEATURES_STRADDLING_ZERO_TITLE)
  }

  override def transform(logical: BootstrapReport): SectionPhysicalReport = {
    new SectionPhysicalReport(
      Seq(
        getMetricsDistribution(logical),
        getBootstrapMetrics(logical),
        getImportantFeatures(logical),
        getStraddlingZeroSection(logical)),
      BOOTSTRAP_SECTION_TITLE)
  }
}

object BootstrapToPhysicalReportTransformer {
  val BOOTSTRAP_SECTION_TITLE = "Bootstrap Analysis"
  val BAGGED_MODEL_METRICS_SECTION_TITLE = "Bagged Model Metrics"
  val METRICS_DISTRIBUTION_SECTION_TITLE = "Metrics Distributions"
  val IMPORTANT_FEATURES_SECTION_TITLE = "Coefficient Analysis for Important Features"
  val FEATURES_STRADDLING_ZERO_TITLE = "Features Straddling Zero"
}
