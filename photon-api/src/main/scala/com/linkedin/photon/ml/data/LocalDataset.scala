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
package com.linkedin.photon.ml.data

/**
 * Local dataset implementation.
 *
 * @note One design concern is whether to store the local data as a [[Map]] or an [[Array]] (high sort cost, but low
 *       merge cost vs. no sort cost but high merge cost). Currently, we use an [[Array]] since the data is only sorted
 *       once, and used as the base for all other data/score [[Array]]s.
 *
 * @param dataPoints Local data points consists of (globalId, labeledPoint) pairs
 */
protected[ml] case class LocalDataset(dataPoints: Array[LabeledPoint]) {

  require(
    dataPoints.length > 0,
    "Cannot create LocalDataset with empty data array")

  val numDataPoints: Int = dataPoints.length
  val numFeatures: Int = dataPoints
    .head
    .features
    .length
}