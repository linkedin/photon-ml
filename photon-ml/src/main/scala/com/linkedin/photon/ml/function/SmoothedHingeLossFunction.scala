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
package com.linkedin.photon.ml.function

import breeze.linalg
import com.linkedin.photon.ml.data.LabeledPoint

/**
 * Implement Rennie's smoothed hinge loss function (http://qwone.com/~jason/writing/smoothHinge.pdf) as an
 * optimizer-friendly approximation for linear SVMs
 */
class SmoothedHingeLossFunction extends DiffFunction[LabeledPoint] {

  override protected[ml] def calculateAt(
      datum: LabeledPoint,
      coefficients: linalg.Vector[Double],
      cumGradient: linalg.Vector[Double]): Double = {

    val actualLabel = if (datum.label < 0.5) -1 else 1
    val margin = datum.computeMargin(coefficients)
    val z = actualLabel * margin

    // Eq: 2, page 2
    val loss = if (z <= 0) {
      0.5 - z
    } else if (z < 1) {
      0.5 * (1.0 - z) * (1.0 - z)
    } else {
      0.0
    }

    // Eq. 3, page 2
    val deriv = if (z < 0) {
      -1.0
    } else if( z < 1) {
      z - 1.0
    } else {
      0.0
    }

    // Eq. 4, page 2
    // cumGradient += datum.weight * actualLabel * deriv * datum.features
    breeze.linalg.axpy(datum.weight * actualLabel * deriv, datum.features, cumGradient)
    datum.weight * loss
  }
}
