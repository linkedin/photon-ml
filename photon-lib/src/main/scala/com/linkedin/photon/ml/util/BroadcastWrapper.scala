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
 * Wrapper to support both Spark [[Broadcast]] and non-broadcast objects.
 *
 * @tparam T Some object type
 */
sealed trait BroadcastWrapper[T] {

  /**
   * Getter for the wrapped object.
   *
   * @return The object
   */
  def value: T

  /**
   * Method used to define equality on multiple class levels while conforming to equality contract. Defines under
   * what circumstances this class can equal another class.
   *
   * @param other Some other object
   * @return True if the object can be compared to this object, false otherwise
   */
  def canEqual(other: Any): Boolean

  /**
   * Compare two objects.
   *
   * @param other Some other object
   * @return True if the object and this object are effectively equivalent, false otherwise
   */
  override def equals(other: Any): Boolean = other match {
    case that: BroadcastWrapper[T] => canEqual(that) && value.equals(that.value)
    case _ => false
  }

  /**
   * Get a hash code for this object.
   *
   * @return The hash code of the wrapped object
   */
  override def hashCode: Int = value.hashCode
}

/**
 * Wrapper around an object that has been broadcast to the Spark executors.
 *
 * @tparam T Some object type
 * @param bv A value in a Spark [[Broadcast]]
 */
case class PhotonBroadcast[T](bv: Broadcast[T]) extends BroadcastWrapper[T] {

  /**
   * Getter for the wrapped object.
   *
   * @return The broadcast object
   */
  def value: T = bv.value

  /**
   * Method used to define equality on multiple class levels while conforming to equality contract. Defines under
   * what circumstances this class can equal another class.
   *
   * @param other Some other object
   * @return True if the object can be compared to this object, false otherwise
   */
  override def canEqual(other: Any): Boolean = other.isInstanceOf[PhotonBroadcast[T]]
}

/**
 * Wrapper around a non-broadcast object.
 *
 * @tparam T Some object type
 * @param value A non-broadcast object
 */
case class PhotonNonBroadcast[T](value: T) extends BroadcastWrapper[T] {

  /**
   * Method used to define equality on multiple class levels while conforming to equality contract. Defines under
   * what circumstances this class can equal another class.
   *
   * @param other Some other object
   * @return True if the object can be compared to this object, false otherwise
   */
  override def canEqual(other: Any): Boolean = other.isInstanceOf[PhotonNonBroadcast[T]]
}