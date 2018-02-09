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
package com.linkedin.photon.ml.diagnostics

import com.linkedin.photon.ml.diagnostics.DiagnosticMode.DiagnosticMode

/**
 * A compact representation of the model diagnostic status.
 *
 * @param trainDiagnosed Whether model training is diagnosed
 * @param validateDiagnosed Whether model  with validation data
 */
// TODO: Remove along with legacy Driver
protected[ml] case class DiagnosticStatus(var trainDiagnosed: Boolean, var validateDiagnosed: Boolean) {

  /**
   * Get the diagnostic mode based on the current diagnostic status.
   *
   * @return The diagnostic mode
   */
  def getDiagnosticMode: DiagnosticMode = {
    if (trainDiagnosed && validateDiagnosed) {
      DiagnosticMode.ALL
    } else if (trainDiagnosed) {
      DiagnosticMode.TRAIN
    } else if (validateDiagnosed) {
      DiagnosticMode.VALIDATE
    } else {
      DiagnosticMode.NONE
    }
  }
}
