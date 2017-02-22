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

import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast

/**
 * A loader that provides instances of DefaultIndexMap
 */
class DefaultIndexMapLoader(sc: SparkContext, featureNameToIdMap: Map[String, Int]) extends IndexMapLoader {

  @transient
  private val _indexMap: IndexMap = new DefaultIndexMap(featureNameToIdMap)
  private val _mapBroadCaster: Broadcast[Map[String, Int]] = sc.broadcast(featureNameToIdMap)

  /**
   *
   * @return The loaded IndexMap for driver
   */
  override def indexMapForDriver(): IndexMap = _indexMap

  /**
   *
   * @return The loaded IndexMap for RDDs
   */
  override def indexMapForRDD(): IndexMap = new DefaultIndexMap(_mapBroadCaster.value)
}

object DefaultIndexMapLoader {

  /**
   * A factory method to create default index map loaders.
   *
   * @note The resulting indexes take their values in [0..numFeatures] ("distinct.sorted.zipWithIndex").
   *
   * @param sc The Spark context
   * @param featureNames A set of feature names
   * @return A default index map loader
   */
  def apply(sc: SparkContext, featureNames: Seq[String]): DefaultIndexMapLoader =
    new DefaultIndexMapLoader(sc, featureNames.distinct.sorted.zipWithIndex.toMap)

  /**
   * A factory method to create a default index map loader from an existing IndexMap, which is essentially
   * the same as a Map[String, Int].
   *
   * @param sc The Spark context
   * @param featureIndex The feature index to wrap in a loader
   * @return A default index map loader
   */
  def apply(sc: SparkContext, featureIndex: IndexMap): DefaultIndexMapLoader =
    new DefaultIndexMapLoader(sc, featureIndex)
}
