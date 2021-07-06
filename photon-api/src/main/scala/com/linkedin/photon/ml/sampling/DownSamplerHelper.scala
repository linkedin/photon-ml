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
package com.linkedin.photon.ml.sampling

import com.linkedin.photon.ml.TaskType
import com.linkedin.photon.ml.TaskType.TaskType

/**
 * Helper for [[DownSampler]] related tasks.
 */
object DownSamplerHelper {

  type DownSamplerFactory = (Double) => DownSampler

  /**
   * Construct a factory function for building [[DownSampler]] objects.
   *
   * @param trainingTask The type of training task being performed on the data which is being down-sampled
   * @return A function which builds a [[DownSampler]] of the appropriate type for a given down-sampling rate
   */
  def buildFactory(trainingTask: TaskType): DownSamplerFactory = trainingTask match {

    case TaskType.LOGISTIC_REGRESSION =>
      (downSamplingRate: Double) => new BinaryClassificationDownSampler(downSamplingRate)

    case TaskType.LINEAR_REGRESSION | TaskType.POISSON_REGRESSION =>
      (downSamplingRate: Double) => new DefaultDownSampler(downSamplingRate)
  }
}
