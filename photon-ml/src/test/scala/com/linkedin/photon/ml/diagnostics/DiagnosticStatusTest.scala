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
package com.linkedin.photon.ml.diagnostics

import org.testng.annotations.Test
import org.testng.Assert.assertEquals

/**
 * Simple tests for [[DiagnosticStatus]]
 */
class DiagnosticStatusTest {
  @Test
  def testGetDiagnosticMode(): Unit = {
    val diagnosticStatus = DiagnosticStatus(trainDiagnosed = false, validateDiagnosed = false)
    assertEquals(diagnosticStatus.getDiagnosticMode, DiagnosticMode.NONE)
    diagnosticStatus.trainDiagnosed = true
    diagnosticStatus.validateDiagnosed = false
    assertEquals(diagnosticStatus.getDiagnosticMode, DiagnosticMode.TRAIN)
    diagnosticStatus.trainDiagnosed = false
    diagnosticStatus.validateDiagnosed = true
    assertEquals(diagnosticStatus.getDiagnosticMode, DiagnosticMode.VALIDATE)
    diagnosticStatus.trainDiagnosed = true
    diagnosticStatus.validateDiagnosed = true
    assertEquals(diagnosticStatus.getDiagnosticMode, DiagnosticMode.ALL)
  }
}
