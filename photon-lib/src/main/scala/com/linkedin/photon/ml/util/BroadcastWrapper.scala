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

import org.apache.spark.broadcast.Broadcast

/**
  * Broadcast wrapper to support both spark Broadcast[T] and non-broadcast variable T.
  * First applied in NormalizationContext.
  * Broadcast[NormalizationContext] in FixedEffect and certain RandomEffect Projection.
  * NormalizationContext in IndexMapProjection RandomEffect.
  */
sealed trait BroadcastWrapper[T] {
  def value: T
}

case class PhotonBroadcast[T](va: Broadcast[T]) extends BroadcastWrapper[T] {
  def value = va.value

  def unpersist(): Unit = { va.unpersist() }
}

case class PhotonNonBroadcast[T](value: T) extends BroadcastWrapper[T]