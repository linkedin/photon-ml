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
package com.linkedin.photon.ml

import com.linkedin.photon.ml.util.Utils

/**
 * Some commonly used String constants.
 */
object Constants {

  /**
   * Delimiter used to concatenate feature name and term into feature key.
   *
   * WARNING: This is not visible in println!
   */
  val DELIMITER = "\u0001"

  /**
   * Wildcard character used for specifying the feature constraints. Only the term is allowed to be a wildcard normally
   * unless one wants to apply bounds to all features in which case both name and term can be specified as wildcards.
   * Currently, we do not support wildcards in name alone.
   */
  val WILDCARD = "*"

  val INTERCEPT_NAME = "(INTERCEPT)"
  val INTERCEPT_TERM = ""
  val INTERCEPT_KEY: String = Utils.getFeatureKey(INTERCEPT_NAME, INTERCEPT_TERM)
}
