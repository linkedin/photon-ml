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
package com.linkedin.photon.ml.function.glm

import com.linkedin.photon.ml.constants.MathConst
import com.linkedin.photon.ml.util.MathUtils

/**
 * Class for the logistic loss function:
 *
 *   sum_i (w_i*(y_i*log(1 + exp(-(theta'x_i + o_i))) + (1-y_i)*log(1 + exp(theta'x_i + o_i)))),
 *
 * where:
 *
 *  - \theta is the vector of estimated coefficient weights for the data features
 *  - (y_i, x_i, o_i, w_i) are the tuple (label, features, offset, weight) of the i'th labeled data point
 *
 * Note that the above equation assumes that:
 *
 *    y_i \in {0, 1}.
 *
 * However, the code below would also work when:
 *
 *    y_i \in {-1, 1}.
 *
 * Logistic regression single loss function:
 *
 * l(z, y) = - log [1 / (1 + exp(-z))]           if this is a positive sample
 *
 *           - log [1 - (1 / (1 + exp(-z)))]     if this is a negative sample
 */
@SerialVersionUID(1L)
object LogisticLossFunction extends PointwiseLossFunction {
  /**
   * The sigmoid function:
   *
   *    1 / (1 + exp(-z))
   *
   * @param z The margin, i.e. z in l(z, y)
   * @return The value
   */
  private def sigmoid(z: Double): Double = 1.0 / (1.0 + math.exp(-z))


  /**
   * l(z, y) = - log [1 / (1 + exp(-z))]       = log [1 + exp(-z)]     if this is a positive sample
   *
   *           - log [1 - (1 / (1 + exp(-z)))] = log [1 + exp(z)]      if this is a negative sample
   *
   * dl/dz   = -1 / (1 + exp(z))         if this is a positive sample
   *
   *           1 / (1 + exp(-z))          if this is a negative sample
   *
   * @param margin The margin, i.e. z in l(z, y)
   * @param label The label, i.e. y in l(z, y)
   * @return The value and the 1st derivative
   */
  override def lossAndDzLoss(margin: Double, label: Double): (Double, Double) = {
    if (label > MathConst.POSITIVE_RESPONSE_THRESHOLD) {
      // The following is equivalent to log(1 + exp(-margin)) but more numerically stable.
      (MathUtils.log1pExp(-margin), -sigmoid(-margin))
    } else {
      (MathUtils.log1pExp(margin), sigmoid(margin))
    }
  }

  /**
   * d^2^l/dz^2^ = sigmoid(z) * (1 - sigmoid(z))
   *
   * @param margin The margin, i.e. z in l(z, y)
   * @param label The label, i.e. y in l(z, y)
   * @return The value and the 2nd derivative with respect to z
   */
  override def DzzLoss(margin: Double, label: Double): Double = {
    val s = sigmoid(margin)
    s * (1 - s)
  }
}
