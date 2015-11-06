package com.linkedin.photon.ml.data


import org.apache.spark.broadcast.Broadcast


/**
 * A wrapper for normal object and broadcast object. This approach simplifies logic for classes/methods that deal with
 * both iterable and RDD data.
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
