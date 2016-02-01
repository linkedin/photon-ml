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


import com.linkedin.photon.ml.data.{SimpleObjectProvider, ObjectProvider}
import com.linkedin.photon.ml.normalization.{NoNormalization, NormalizationContext}


/**
 * Class for the Poisson loss function: sum_i (w_i*(exp(theta'x_i + o_i) - y_i*(theta'x_i + o_i))),
 * where \theta is the coefficients of the data features to be estimated, (y_i, x_i, o_i, w_i) are the tuple
 * for label, features, offset, and weight of the i'th labeled data point, respectively.
 * @author asaha
 * @author dpeng
 */
class PoissonLossFunction(normalizationContext: ObjectProvider[NormalizationContext] = new SimpleObjectProvider[NormalizationContext](NoNormalization)) extends
  GeneralizedLinearModelLossFunction(PointwisePoissonLossFunction, normalizationContext)

/**
 * Poisson regression single loss function
 *
 * l(z, y) = exp(z) - y * z
 *
 * Used for Poisson regression
 */
@SerialVersionUID(1L)
object PointwisePoissonLossFunction extends PointwiseLossFunction {
  /**
   * l(z, y) = exp(z) - y * z
   *
   * dl/dz = exp(z) - y
   * @param margin The margin, i.e. z in l(z, y)
   * @param label The label, i.e. y in l(z, y)
   * @return The value and the 1st derivative
   */
  override def loss(margin: Double, label: Double): (Double, Double) = {
    val prediction = math.exp(margin)
    (prediction - margin * label, prediction - label)
  }

  /**
   * d^2^l/dz^2^ = exp(z)
   * @param margin The margin, i.e. z in l(z, y)
   * @param label The label, i.e. y in l(z, y)
   * @return The value and the 2st derivative with respect to z
   */
  override def d2lossdz2(margin: Double, label: Double): Double = {
    math.exp(margin)
  }
}
