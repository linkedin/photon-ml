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

import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.Identifiable

/**
 * Utility functions used for working with [[Param]] objects.
 */
object ParamUtils {

  /**
   * Create a new [[Param]].
   *
   * @tparam T The type of the parameter
   * @param name The name of the parameter
   * @param doc The documentation for the parameter
   * @param isValid The function used to verify whether an input value is a valid argument
   * @param parent The implicit owner object of the parameter
   * @return A new [[Param]]
   */
  protected[ml] def createParam[T]
      (name: String, doc: String, isValid: T => Boolean)
      (implicit parent: Identifiable): Param[T] =
    new Param(parent, name, doc, isValid)

  /**
   * Create a new [[Param]] which accepts any argument as valid.
   *
   * @tparam T The type of the parameter
   * @param name The name of the parameter
   * @param doc The documentation for the parameter
   * @param parent The implicit owner object of the parameter
   * @return A new [[Param]]
   */
  protected[ml] def createParam[T](name: String, doc: String)(implicit parent: Identifiable): Param[T] =
    new Param(parent, name, doc)
}
