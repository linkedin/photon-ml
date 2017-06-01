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
package com.linkedin.photon.ml.constants

/**
 * Math constants.
 */
object MathConst {
  protected[ml] val EPSILON: Double = 1e-12
  protected[ml] val RANDOM_SEED: Long = 1234567890L
  protected[ml] val POSITIVE_RESPONSE_THRESHOLD = 0.5
  protected[ml] val DEFAULT_WEIGHT = 1.0
  protected[ml] val DEFAULT_OFFSET = 0.0
}
