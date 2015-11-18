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
package com.linkedin.photon.ml.diagnostics.hl

import com.linkedin.photon.ml.diagnostics.reporting.{BulletedListPhysicalReport, PlotPhysicalReport, SectionPhysicalReport, PhysicalReport}
import org.testng.annotations.Test

/**
 * "Marble roll" test of transformer -- verify structure but not contents of output (this should ultimately be inspected
 * by a human to figure out whether this is "correct" or not)
 */
class NaiveHosmerLemeshowToPhysicalReportTransformerTest {

  import org.testng.Assert._

  private def fakeHLReport(): HosmerLemeshowReport = {
    val chiSquareMsg = "This is a fake message representing what should show up in the description of how the chi%2 score was computed"
    val binMsg = "This is a fake message representing what should show up in the description of how the binning works"
    val chiSquareProb = 0.0
    val chiSquareDist = 0.0
    val dof = 20
    val histogram = HosmerLemeshowDiagnosticTest.generatePerfect(dof + 2)
    val cutoffs = List((10.0, 0.1), (20.0, 0.2), (30.0, 0.3))
    new HosmerLemeshowReport(binMsg, chiSquareMsg, chiSquareDist, dof, chiSquareProb, cutoffs, histogram)
  }

  @Test
  def checkReport(): Unit = {
    val report = fakeHLReport()
    val transformer = new NaiveHosmerLemeshowToPhysicalReportTransformer()
    val transformed = transformer.transform(report)
    checkTransformed(transformed)
  }

  private def checkTransformed(report: PhysicalReport): Unit = {
    report match {
      case sec: SectionPhysicalReport =>
        assertEquals(sec.items.length, 2, "Expected number of sections")
        assertEquals(sec.title, NaiveHosmerLemeshowToPhysicalReportTransformer.SECTION_HEADER)

        sec.items(0) match {
          case plotSec: SectionPhysicalReport =>
            checkPlotSection(plotSec)
          case _ =>
            fail(s"First item in section should be a subsection, got a ${sec.items(0).getClass.getName}")
        }

        sec.items(1) match {
          case analysisSec: SectionPhysicalReport =>
            checkAnalysisSection(analysisSec)
          case _ =>
            fail(s"First item in section should be a subsection, got a ${sec.items(1).getClass.getName}")

        }

      case _ =>
        fail(s"Report has wrong type (expect SectionPhysicalReport, got ${report.getClass.getName}")
    }
  }

  private def checkPlotSection(sec: SectionPhysicalReport): Unit = {
    assertEquals(sec.title, NaiveHosmerLemeshowToPhysicalReportTransformer.PLOT_HEADER)
    assertEquals(sec.items.length, 4)
    sec.items.foreach(x => assertTrue(x.isInstanceOf[PlotPhysicalReport]))
  }

  private def checkAnalysisSection(sec: SectionPhysicalReport): Unit = {
    assertEquals(sec.title, NaiveHosmerLemeshowToPhysicalReportTransformer.ANALYSIS_HEADER)
    assertEquals(sec.items.length, 3)
    assertTrue(sec.items(0).isInstanceOf[BulletedListPhysicalReport])
    assertTrue(sec.items(1).isInstanceOf[SectionPhysicalReport])
    assertTrue(sec.items(2).isInstanceOf[SectionPhysicalReport])
  }
}
