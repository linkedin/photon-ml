/*
 * Copyright 2014 LinkedIn Corp. All rights reserved.
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
package com.linkedin.photon.ml.diagnostics.reporting.reports.combined

import java.text.SimpleDateFormat
import java.util.{Date, TimeZone, Calendar}

import com.linkedin.photon.ml.diagnostics.reporting.{ChapterPhysicalReport, DocumentPhysicalReport, LogicalToPhysicalReportTransformer}
import com.linkedin.photon.ml.diagnostics.reporting.reports.model.ModelDiagnosticToPhysicalReportTransformer
import com.linkedin.photon.ml.diagnostics.reporting.reports.system.SystemToPhysicalReportTransformer
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel

/**
 * Transform diagnostic reports into their physical report representation
 */
class DiagnosticToPhysicalReportTransformer extends LogicalToPhysicalReportTransformer[DiagnosticReport, DocumentPhysicalReport] {

  import DiagnosticToPhysicalReportTransformer._

  def transform(diag: DiagnosticReport): DocumentPhysicalReport = {
    val formatter = new SimpleDateFormat()
    val now = new Date()
    new DocumentPhysicalReport(
      Seq(
        SYSTEM_CHAPTER_TRANSFORMER.transform(diag.systemReport),
        new ChapterPhysicalReport
        (diag.modelReports.toArray.sortBy(x => x.lambda).map(x => MODEL_SECTION_TRANSFORMER.transform(x)).toSeq, MODEL_CHAPTER_TITLE)),
      s"Modeling run ${formatter.format(now)}")
  }
}

object DiagnosticToPhysicalReportTransformer {
  val SYSTEM_CHAPTER_TRANSFORMER = new SystemToPhysicalReportTransformer()
  val MODEL_SECTION_TRANSFORMER = new ModelDiagnosticToPhysicalReportTransformer[GeneralizedLinearModel]()
  val MODEL_CHAPTER_TITLE = "Detailed Model Diagnostics"
}