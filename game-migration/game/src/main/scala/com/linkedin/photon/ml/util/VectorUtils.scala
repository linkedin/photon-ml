package com.linkedin.photon.ml.util


import scala.collection.mutable

import breeze.linalg.{DenseVector, SparseVector, Vector}


/**
 * @author xazhang
 */
protected[ml] object VectorUtils {

  val SPARSE_VECTOR_ACTIVE_SIZE_TO_SIZE_RATIO: Double = 1.0 / 3

  def indexAndValueArrayToVector(
      indexAndData: Array[(Int, Double)],
      length: Int,
      sparseVectorActiveSizeToSizeRatio: Double = SPARSE_VECTOR_ACTIVE_SIZE_TO_SIZE_RATIO): Vector[Double] = {
    if (length * SPARSE_VECTOR_ACTIVE_SIZE_TO_SIZE_RATIO < indexAndData.length) {
      indexAndValueArrayToDenseVector(indexAndData, length)
    } else {
      indexAndValueArrayToSparseVector(indexAndData, length)
    }
  }

  def indexAndValueArrayToSparseVector(indexAndData: Array[(Int, Double)], length: Int): SparseVector[Double] = {
    val sortedIndexAndData = indexAndData.sortBy(_._1)
    val index = new Array[Int](sortedIndexAndData.length)
    val data = new Array[Double](sortedIndexAndData.length)
    var i = 0
    while (i < sortedIndexAndData.length) {
      index(i) = sortedIndexAndData(i)._1
      data(i) = sortedIndexAndData(i)._2
      i += 1
    }
    new SparseVector[Double](index, data, length)
  }

  def indexAndValueArrayToDenseVector(indexAndData: Array[(Int, Double)], length: Int): DenseVector[Double] = {
    val dataArray = new Array[Double](length)
    var i = 0
    while (i < indexAndData.length) {
      val (index, data) = indexAndData(i)
      dataArray(index) = data
      i += 1
    }
    new DenseVector[Double](dataArray)
  }

  /**
   * The Kronecker product between two vectors: vector1 \otimes vector2
   * @param vector1 The left vector
   * @param vector2 The right vector
   * @param threshold Threshold of the cross value
   * @return The resulting Kronecker product between vector1 and vector2
   */
  def kroneckerProduct(vector1: Vector[Double], vector2: Vector[Double], threshold: Double): Vector[Double] = {

    assert(vector1.isInstanceOf[SparseVector[Double]] || vector2.isInstanceOf[SparseVector[Double]],
      "Kronecker product between two dense vectors is currently not supported!")

    val length = vector1.length * vector2.length
    val activeSize = vector1.activeSize * vector2.activeSize
    val index = new mutable.ArrayBuffer[Int](activeSize)
    val data = new mutable.ArrayBuffer[Double](activeSize)
    for ( (idx1, data1) <- vector1.activeIterator) {
      for ( (idx2, data2) <- vector2.activeIterator ) {
        val crossedValue = data1 * data2
        if (math.abs(crossedValue) > threshold) {
          val idx = idx1 * vector2.length + idx2
          index += idx
          data += crossedValue
        }
      }
    }
    new SparseVector[Double](index.toArray, data.toArray, length)
  }
}
