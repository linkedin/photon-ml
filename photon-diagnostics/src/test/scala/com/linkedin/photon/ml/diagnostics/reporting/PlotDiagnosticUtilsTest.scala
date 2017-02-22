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
package com.linkedin.photon.ml.diagnostics.reporting

import com.linkedin.photon.ml.Evaluation
import org.testng.annotations._

/**
 * Check PlotUtils public methods
 */
class PlotDiagnosticUtilsTest {
  import org.testng.Assert._

  @Test
  def checkMetricsWithRanges(): Unit = {
    val toCheck = Evaluation.metricMetadata.filter(_._2.rangeOption.isDefined).toSeq
    val values = Seq(-1e3, 1e3)

    toCheck.foreach( x => {
      val metric = x._1
      val expectedRange = x._2.rangeOption.get
      val computedRange = PlotUtils.getRangeForMetric(metric, values)
      assertEquals(computedRange._1, expectedRange._1)
      assertEquals(computedRange._2, expectedRange._2)

    })
  }

  @Test
  def checkMetricsWithoutRanges(): Unit = {
    val toCheck = Evaluation.metricMetadata.filter(_._2.rangeOption.isEmpty).toSeq
    val values = Seq(-1e3, 1e3)

    toCheck.foreach( x => {
      val metric = x._1
      val expectedRange = Seq(-1020.0, 1020.0)
      val computedRange = PlotUtils.getRangeForMetric(metric, values)
      assertEquals(computedRange._1, expectedRange(0))
      assertEquals(computedRange._2, expectedRange(1))
    })
  }

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def checkInvalidMetric(): Unit = {
    PlotUtils.getRangeForMetric("THIS IS A FAKE METRIC", null)
  }

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def checkNonFinite(): Unit = {
    PlotUtils.getRange(Seq(Double.NaN))
  }
}
