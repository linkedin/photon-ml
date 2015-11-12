/*
 * Copyright 2014 LinkedIn Corp. All rights reserved.
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


import org.apache.spark.broadcast.Broadcast


/**
 * A wrapper for normal object and broadcast object. This approach simplifies logic for classes/methods that deal with
 * both iterable and RDD data.
 *
 * This trait hides the logic whether the data are local or from remote. Probably it will be better if Iterable and
 * RDD data are separated out in [[com.linkedin.photon.ml.function.DiffFunction]] and [[com.linkedin.photon.ml.function.TwiceDiffFunction]]
 * classes.
 *
 * @author dpeng
 */
trait ObjectProvider[T <: Serializable] extends Serializable {
  def get: T
}

@SerialVersionUID(1L)
class SimpleObjectProvider[T <: Serializable](obj: T) extends ObjectProvider[T] {
  override def get: T = obj
}

@SerialVersionUID(1L)
class BroadcastedObjectProvider[T <: Serializable](obj: Broadcast[T]) extends ObjectProvider[T] {
  override def get: T = obj.value
}
