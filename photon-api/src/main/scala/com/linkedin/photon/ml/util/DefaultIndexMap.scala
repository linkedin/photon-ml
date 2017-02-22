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
package com.linkedin.photon.ml.util

/**
 * Use the default system HashMap to construct an index map. Highly inefficient in terms of memory usage, but easier to
 * handle.
 *
 * Recommended for small feature space cases (<= 200k).
 *
 * For use cases where number of features > 200k, please try [[PalDBIndexMap]] instead.
 *
 * @param featureNameToIdMap The map from raw feature string (name) to feature index (id)
 */
class DefaultIndexMap(@transient val featureNameToIdMap: Map[String, Int]) extends IndexMap {
  @transient
  private var _idToNameMap: Map[Int, String] = _
  private val _size: Int = featureNameToIdMap.size

  override def size(): Int = _size
  override def isEmpty(): Boolean = size == 0
  override def getFeatureName(idx: Int): Option[String] = {
    if (_idToNameMap == null) {
      _idToNameMap = featureNameToIdMap.map{case (k, v) => (v, k)}
    }

    _idToNameMap.get(idx)
  }
  override def getIndex(name: String): Int = featureNameToIdMap.getOrElse(name, IndexMap.NULL_KEY)
  override def +[B1 >: Int](kv: (String, B1)): Map[String, B1] = featureNameToIdMap.+(kv)
  override def get(key: String): Option[Int] = featureNameToIdMap.get(key)
  override def iterator: Iterator[(String, Int)] = featureNameToIdMap.iterator
  override def -(key: String): Map[String, Int] = featureNameToIdMap.-(key)
}

object DefaultIndexMap {
  /**
   * Factory to build default index maps from sets of feature names.
   *
   * NOTE: the resulting indexes take their values in [0..numFeatures] ("distinct.sorted.zipWithIndex").
   *
   * @param featureNames The feature names we need in the index map
   * @return A DefaultIndexMap instance for these feature names
   */
  def apply(featureNames: Seq[String]): DefaultIndexMap = {
    new DefaultIndexMap(featureNames.distinct.sorted.zipWithIndex.toMap)
  }
}
