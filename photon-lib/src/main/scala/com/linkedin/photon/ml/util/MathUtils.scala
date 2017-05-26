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
package com.linkedin.photon.ml.util

import com.linkedin.photon.ml.constants.MathConst

/**
 *
 */
object MathUtils {

  /**
   * This function is copied from MLlib's MLUtils.log1pExp (it is copied instead of imported because it is private).
   *
   * When `x` is positive and large, computing `math.log(1 + math.exp(x))` will lead to arithmetic overflow. This will
   * happen when `x > 709.78` which is not a very large number. It can be addressed by rewriting the formula into
   * `x + math.log1p(math.exp(-x))` when `x > 0`.
   *
   * @param x A floating-point value as input.
   * @return The result of `math.log(1 + math.exp(x))`.
   */
  def log1pExp(x: Double): Double = {
    if (x > 0) {
      x + math.log1p(math.exp(-x))
    } else {
      math.log1p(math.exp(x))
    }
  }

  /**
   * Decides when a very small value should be considered equal to zero.
   *
   * @param x The value to test for near-equality to zero
   * @return True if x is "as good as" zero, false if it is "significantly" different from zero
   */
  def isAlmostZero(x: Double): Boolean = math.abs(x) < MathConst.EPSILON
}
