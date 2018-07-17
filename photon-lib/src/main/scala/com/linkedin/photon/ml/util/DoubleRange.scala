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

/**
 * Represents an immutable numeric range.
 *
 * @param start The beginning of the range
 * @param end The end of the range
 */
case class DoubleRange(start: Double, end: Double) {

  require(start <= end, s"Invalid range: start $start comes after end date $end.")

  /**
   * Applies the transformation function to the range
   *
   * @param fn The transformation function
   * @return The transformed range
   */
  def transform(fn: Double => Double): DoubleRange = DoubleRange(fn(start), fn(end))
}
