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

/**
 * A feature index map backed by an immutable Map.
 * Highly inefficient in terms of memory usage, but easier to handle.
 *
 * Recommended for small feature space cases (<= 200k).
 *
 * For use cases where number of features > 200k, please try [[PalDBIndexMap]] instead.
 *
 * @param featureNameToIdMap The map from raw feature string (name) to feature index (id)
 */
class DefaultIndexMap(val featureNameToIdMap: Map[String, Int]) extends IndexMap {

  private var _idToNameMap: Map[Int, String] = _
  private val _size: Int = featureNameToIdMap.size

  override def size(): Int = _size
  override def isEmpty(): Boolean = size == 0

  /**
   * This function lazily creates _idToNameMap when called for the first time.
   *
   * @param idx The feature index
   * @return The feature name if found, NONE otherwise
   */
  override def getFeatureName(idx: Int): Option[String] = {
    if (_idToNameMap == null) {
      _idToNameMap = featureNameToIdMap.map{case (k, v) => (v, k)}
    }

    _idToNameMap.get(idx)
  }

  /**
   * Get an Option containing the feature id for a given feature name, or None if there is no such feature.
   *
   * @param key The feature name
   * @return Some(feature index) if the feature exists, None otherwise
   */
  override def get(key: String): Option[Int] = featureNameToIdMap.get(key)

  /**
   * Get a feature index from a feature name.
   *
   * @param name The feature name
   * @return The feature index if found, IndexMap.NULL_KEY otherwise
   */
  override def getIndex(name: String): Int = featureNameToIdMap.getOrElse(name, IndexMap.NULL_KEY)

  /**
   * Get an iterator over all the (feature name, feature index) pairs.
   *
   * @return The iterator
   */
  override def iterator: Iterator[(String, Int)] = featureNameToIdMap.iterator

  def +[B >: Int](kv: (String, B)): DefaultIndexMap =
    new DefaultIndexMap(featureNameToIdMap. +(kv).asInstanceOf[Map[String, Int]])
  def -(key: String): DefaultIndexMap = new DefaultIndexMap(featureNameToIdMap. -(key))
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
