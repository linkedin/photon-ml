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

import scala.collection.immutable.Map

/**
 * The trait defines the methods supposed should be supported by a feature index map.
 */
trait IndexMap extends Map[String, Int] with Serializable {

  /**
   * Lazily compute and cache the number of features in this index
   */
  lazy val featureDimension: Int = values.max + 1

  /**
   * Given an index, return the corresponding feature name.
   *
   * @param idx The feature index
   * @return The feature name if found, NONE otherwise
   */
  def getFeatureName(idx: Int): Option[String]

  /**
   * Given a feature name, return its index.
   *
   * @param name The feature name
   * @return The feature index if found, IndexMap.NULL_KEY otherwise
   */
  def getIndex(name: String): Int
}

object IndexMap {

  // The key to indicate a feature does not exist in the map
  val NULL_KEY: Int = -1

  // "global" namespace for situations where either there aren't multiple namespaces, or we want to set apart a global
  // namespace from subspaces
  val GLOBAL_NS: String = "global"
}
