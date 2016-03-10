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
package com.linkedin.photon.ml.constants

/**
 * Math constants
 *
 * @author xazhang
 */
object MathConst {
  val HIGH_PRECISION_TOLERANCE_THRESHOLD: Double = 1e-12
  val MEDIUM_PRECISION_TOLERANCE_THRESHOLD: Double = 1e-8
  val LOW_PRECISION_TOLERANCE_THRESHOLD: Double = 1e-4
  val RANDOM_SEED: Long = 1234567890L
  val POSITIVE_RESPONSE_THRESHOLD = 0.5
}
