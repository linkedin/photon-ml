package com.linkedin.photon.ml.data

import java.{util => Jutil}

import scala.collection.JavaConversions._
import scala.collection.mutable
import scala.reflect.ClassTag

/**
 * Implementation of a min heap datastructure that defers heapification until the given buffer capacity has been
 * reached
 *
 * @param capacity buffer capacity
 * @author xazhang
 */
class MinHeapWithFixedCapacity[T <: Comparable[T] : ClassTag](capacity: Int) extends Serializable {

  private val arrayBuffer = new mutable.ArrayBuffer[T](2)
  var cumCount = 0
  private var minHeap: Jutil.PriorityQueue[T] = _

  /**
   * Add value to the heap
   *
   * @param value the value to add
   * @return the updated heap
   */
  def +=(value: T): this.type = {
    if (cumCount < capacity) {
      arrayBuffer.add(value)
    } else {
      if (minHeap == null) activateMinHeap()
      if (minHeap.peek().compareTo(value) <= 0) {
        minHeap.poll()
        minHeap.offer(value)
      }
    }
    cumCount += 1
    if (cumCount == capacity) {
      activateMinHeap()
    }
    this
  }

  /**
   * Add contents of another heap to the heap
   *
   * @param minHeapWithFixedCapacity the other heap
   * @return the updated heap
   */
  def ++=(minHeapWithFixedCapacity: MinHeapWithFixedCapacity[T]): this.type = {
    if (cumCount + minHeapWithFixedCapacity.cumCount < capacity) {
      arrayBuffer.addAll(minHeapWithFixedCapacity.arrayBuffer)
    } else {
      if (minHeap == null) activateMinHeap()
      val thisHeap = minHeap
      if (minHeapWithFixedCapacity.minHeap == null) minHeapWithFixedCapacity.activateMinHeap()
      val thatHeap = minHeapWithFixedCapacity.minHeap
      while (thisHeap.size() + thatHeap.size() > capacity) {
        if (thisHeap.peek().compareTo(thatHeap.peek()) <= 0) {
          thisHeap.poll()
        }
        else {
          thatHeap.poll()
        }
      }
      val minSize = math.min(thisHeap.size(), thatHeap.size())
      val maxSize = math.max(thisHeap.size(), thatHeap.size())
      if (capacity < minSize * (31 - Integer.numberOfLeadingZeros(maxSize))) {
        val thisArray = thisHeap.toArray.asInstanceOf[Array[T]]
        val thatArray = thatHeap.toArray.asInstanceOf[Array[T]]
        val newArray = new Array[T](capacity)
        Array.copy(thisArray, 0, newArray, 0, thisArray.length)
        Array.copy(thatArray, 0, newArray, thisArray.length, thatArray.length)
        thatHeap.clear()
        thisHeap.clear()
        minHeap = new Jutil.PriorityQueue(newArray.toSeq)
      } else {
        if (thisHeap.size() < thatHeap.size()) {
          thatHeap.addAll(thisHeap)
          thisHeap.clear()
          minHeap = thatHeap
        } else {
          thisHeap.addAll(thatHeap)
          thatHeap.clear()
          minHeap = thisHeap
        }
      }
    }
    cumCount += minHeapWithFixedCapacity.cumCount
    this
  }

  /**
   * Activate (i.e. heapify) the heap
   */
  private def activateMinHeap() = {
    minHeap = new Jutil.PriorityQueue(arrayBuffer)
    arrayBuffer.clear()
  }

  /**
   * Return iterable of the heap data
   *
   * @return heap data
   */
  def getData: Iterable[T] = {
    if (cumCount < capacity) arrayBuffer
    else minHeap
  }
}
