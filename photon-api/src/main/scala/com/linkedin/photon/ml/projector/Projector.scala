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
package com.linkedin.photon.ml.projector

/**
 * Trait for an object which performs two types of projections:
 *   1. Project an object from the original space to the projected space
 *   2. Project an object from the projected space to the original space
 *
 * @tparam T Object type to project between spaces
 */
protected[ml] trait Projector[T] extends Serializable {

  /**
   * Project an object from the original space to the projected space.
   *
   * @param input An input object in the original space
   * @return The same object in the projected space
   */
  def projectForward(input: T): T

  /**
   * Project an object from the projected space back to the original space.
   *
   * @param input An input object in the projected space
   * @return The same object in the original space
   */
  def projectBackward(input: T): T
}
