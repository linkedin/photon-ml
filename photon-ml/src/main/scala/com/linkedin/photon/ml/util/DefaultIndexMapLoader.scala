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
 */
class DefaultIndexMapLoader(@transient val featureNameToIdMap: Map[String, Int]) extends IndexMapLoader {
  @transient
  private var _indexMap: IndexMap = null

  private var _mapBroadCaster: Broadcast[Map[String, Int]] = null

  /**
   * Prepare a loader, should be called early before anything
   */
  override def prepare(sc: SparkContext, params: IndexMapParams, namespace: String): Unit = {
    // do nothing
    _indexMap = new DefaultIndexMap(featureNameToIdMap)
    _mapBroadCaster = sc.broadcast(featureNameToIdMap)
  }

  override def indexMapForDriver(): IndexMap = _indexMap

  override def indexMapForRDD(): IndexMap = new DefaultIndexMap(_mapBroadCaster.value)
}
