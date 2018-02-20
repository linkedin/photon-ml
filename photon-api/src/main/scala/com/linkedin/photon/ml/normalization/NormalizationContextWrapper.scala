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
package com.linkedin.photon.ml.normalization

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.Types.REId

/**
 * Wrapper around [[NormalizationContext]] data.
 */
protected[ml] sealed trait NormalizationContextWrapper {

  /**
   * Remove normalization context data from executor memory and disk.
   */
  def unpersist(): Unit
}

/**
 * Wrapper around a [[RDD]] of unique [[NormalizationContext]] objects.
 *
 * @param contexts Multiple [[NormalizationContext]] objects, one per unique random effect entity.
 */
protected[ml] case class NormalizationContextRDD(contexts: RDD[(REId, NormalizationContext)])
  extends NormalizationContextWrapper {

  /**
   * Mark the RDD as non-persistent, and remove all blocks for it from memory and disk.
   */
  override def unpersist(): Unit = contexts.unpersist()
}

/**
 * Wrapper around a [[Broadcast]] [[NormalizationContext]] object.
 *
 * @param context One [[NormalizationContext]], for training a single distributed optimization problem or for shared use
 *                by multiple optimization problems.
 */
protected[ml] case class NormalizationContextBroadcast(context: Broadcast[NormalizationContext])
  extends NormalizationContextWrapper {

  /**
   * Asynchronously delete cached copies of this broadcast on the executors.
   */
  override def unpersist(): Unit = context.unpersist()
}
