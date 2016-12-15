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

import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast

/**
 * A loader that provides instances of DefaultIndexMap
 *
 * NOTE: instances of this class cannot be used before "prepare" has been called with a SparkContext!
 */
class DefaultIndexMapLoader(@transient val featureNameToIdMap: Map[String, Int]) extends IndexMapLoader {
  @transient
  private var _indexMap: IndexMap = _
  private var _mapBroadCaster: Broadcast[Map[String, Int]] = _

  /**
   * Prepare a loader, should be called early before anything
   */
  override def prepare(
      sc: SparkContext,
      params: IndexMapParams = null,
      namespace: String = IndexMap.GLOBAL_NS): Unit = {

    _indexMap = new DefaultIndexMap(featureNameToIdMap)
    _mapBroadCaster = sc.broadcast(featureNameToIdMap)
  }

  override def indexMapForDriver(): IndexMap = _indexMap
  override def indexMapForRDD(): IndexMap = new DefaultIndexMap(_mapBroadCaster.value)
}

object DefaultIndexMapLoader {

  /**
   * A factory method to create default index map loaders.
   *
   * NOTE: the resulting indexes take their values in [0..numFeatures] ("distinct.sorted.zipWithIndex").
   *
   * @param sc The Spark context
   * @param featureNames A set of feature names
   * @return A default index map loader
   */
  def apply(sc: SparkContext, featureNames: Seq[String]): DefaultIndexMapLoader = {
    // TODO: "distinct.sorted.zipWithIndex" is duplicated in DefaultIndexMap. Remove one.
    val indexMapLoader = new DefaultIndexMapLoader(featureNames.distinct.sorted.zipWithIndex.toMap)
    indexMapLoader.prepare(sc) // need to call this here to complete initialization
    indexMapLoader
  }

  /**
   * A factory method to create a default index map loader from an existing IndexMap, which is essentially
   * the same as a Map[String, Int].
   *
   * @param sc The Spark context
   * @param featureIndex The feature index to wrap in a loader
   * @return A default index map loader
   */
  def apply(sc: SparkContext, featureIndex: IndexMap): DefaultIndexMapLoader = {
    val indexMapLoader = new DefaultIndexMapLoader(featureIndex)
    indexMapLoader.prepare(sc) // need to call this here to complete initialization
    indexMapLoader
  }
}
