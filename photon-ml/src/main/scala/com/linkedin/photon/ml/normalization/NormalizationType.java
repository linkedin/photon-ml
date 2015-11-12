/*
 * Copyright 2014 LinkedIn Corp. All rights reserved.
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
package com.linkedin.photon.ml.normalization;

/**
 * The enum of different normalization types used for feature normalization.
 * @author dpeng
 */
public enum NormalizationType {

  /**
   * Scale each feature to have unit variance
   */
  SCALE_WITH_STANDARD_DEVIATION,

  /**
   * Scale each feature to lie in the range [-1, 1]
   */
  SCALE_WITH_MAX_MAGNITUDE,

  /**
   * Zero-mean unit variance distributions x -> (x - \mu)/\sigma. Intercept must be included to enable this feature.
   */
  STANDARDIZATION,

  /**
   * Skip normalization
   */
  NONE
}
