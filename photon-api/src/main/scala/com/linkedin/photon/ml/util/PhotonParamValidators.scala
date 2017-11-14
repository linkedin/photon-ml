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

import scala.language.higherKinds

/**
 * Custom validators for [[org.apache.spark.ml.param.Param]] used by Photon.
 */
object PhotonParamValidators {

  /**
   * Test whether a collection contains any elements.
   *
   * @tparam Coll The type of the collection
   * @tparam T The type of the elements within the collection
   * @param ev An implicit val to enforce that the collection type derives from [[TraversableOnce]]
   * @return Whether the collection is empty or not
   */
  def nonEmpty[Coll[_], T](implicit ev: Coll[T] <:< TraversableOnce[T]): Coll[T] => Boolean =
    (coll: Coll[T]) => coll.nonEmpty
}
