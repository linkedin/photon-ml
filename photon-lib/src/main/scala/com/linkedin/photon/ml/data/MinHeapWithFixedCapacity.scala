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

import java.{util => Jutil}

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.reflect.ClassTag

/**
 * Implementation of a min heap data structure that defers 'heapification' until the given buffer capacity has been
 * reached.
 *
 * @param capacity Buffer capacity
 */
protected[ml] class MinHeapWithFixedCapacity[T <: Comparable[T] : ClassTag](capacity: Int) extends Serializable {

  require(capacity > 0, s"Invalid heap capacity $capacity; capacity must be greater than 0")

  // Buffer to use until capacity is reached
  // TODO: Better size hint for buffer?
  private val arrayBuffer = new mutable.ArrayBuffer[T](2)
  // Min heap to use once buffer size exceeded
  private var minHeap: Jutil.PriorityQueue[T] = _
  // The number of items that have been attempted to be added to the heap
  private var cumCount = 0

  /**
   * Get the total number of items which have been attempted to be added to the min heap (note that this may be more
   * than the number of items currently in the min heap, since it has fixed capacity).
   *
   * @return The number of items added to the min heap
   */
  def getCount: Int = cumCount

  /**
   * Return the heap data.
   *
   * @return The heap data
   */
  def getData: Iterable[T] =
    if (cumCount < capacity) {
      arrayBuffer
    } else {
      minHeap.asScala
    }

  /**
   * Add a value to the heap.
   *
   * @param value The value to add
   * @return The updated heap
   */
  def +=(value: T): this.type = {

    // If capacity not reached, add to buffer
    if (cumCount < capacity) {
      arrayBuffer += value

    // Otherwise: if new item greater than smallest item in heap, replace the heap item with the new item
    } else if (minHeap.peek().compareTo(value) <= 0) {
      minHeap.poll()
      minHeap.offer(value)
    }

    cumCount += 1
    if (cumCount == capacity) {
      activateMinHeap()
    }

    this
  }

  /**
   * Add contents of another heap to the heap.
   *
   * Warning: This implementation is not thread-safe.
   *
   * @param minHeapWithFixedCapacity The other heap
   * @return The updated heap
   */
  def ++=(minHeapWithFixedCapacity: MinHeapWithFixedCapacity[T]): this.type = {

    cumCount += minHeapWithFixedCapacity.cumCount

    // If combined count doesn't reach capacity, neither heap active yet
    if (cumCount < capacity) {
      arrayBuffer ++= minHeapWithFixedCapacity.arrayBuffer

    } else {
      // Activate heaps, if not yet activated
      if (minHeap == null) activateMinHeap()
      if (minHeapWithFixedCapacity.minHeap == null) minHeapWithFixedCapacity.activateMinHeap()

      val thisHeap = minHeap
      val thatHeap = minHeapWithFixedCapacity.minHeap

      // Empty out heaps to combined size of capacity
      while (thisHeap.size() + thatHeap.size() > capacity) {
        if (thisHeap.peek().compareTo(thatHeap.peek()) <= 0) {
          thisHeap.poll()
        } else {
          thatHeap.poll()
        }
      }

      // Determine relative size of heaps
      val (smallHeap, bigHeap) = if (thisHeap.size() < thatHeap.size()) {
        (thisHeap, thatHeap)
      } else {
        (thatHeap, thisHeap)
      }

      // Determine which strategy to use for merging heaps based on their size:
      //
      // K = heap capacity
      // N = size of smaller heap
      // M = size of larger heap
      //
      // N + M = K, N < M
      //
      // Strategy 1: Copy both and re-heap
      //    N + M + K = 2 * K
      //
      // Strategy 2: Add elements from smaller heap to larger heap
      //    X
      //  such that:
      //    N * log(M) < X < N * log(K)
      //
      // When to use strategy 1:
      //    2 * K < N * log(M) =>
      //    K < N * log(M) / 2 =>
      //    K < X / 2
      //
      if (capacity < smallHeap.size() * (31 - Integer.numberOfLeadingZeros(bigHeap.size())) / 2D) {

        val thisArray = thisHeap.toArray.asInstanceOf[Array[T]]
        val thatArray = thatHeap.toArray.asInstanceOf[Array[T]]
        val newArray = new Array[T](capacity)

        Array.copy(thisArray, 0, newArray, 0, thisArray.length)
        Array.copy(thatArray, 0, newArray, thisArray.length, thatArray.length)
        thisHeap.clear()
        thatHeap.clear()

        minHeap = new Jutil.PriorityQueue[T](newArray.toSet.asJava)

      // Check failed, use strategy 2
      } else {

        bigHeap.addAll(smallHeap)
        bigHeap.clear()
        minHeap = bigHeap
      }
    }

    this
  }

  /**
   * Activate the heap (i.e. heapify the Array).
   */
  private def activateMinHeap(): Unit = {

    minHeap = new Jutil.PriorityQueue[T](arrayBuffer.asJava)
    arrayBuffer.clear()
  }
}
