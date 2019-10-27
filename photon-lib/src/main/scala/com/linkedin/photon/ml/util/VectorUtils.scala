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

import scala.collection.mutable

import breeze.linalg.{DenseVector, SparseVector, Vector}
import org.apache.spark.ml.linalg.{DenseVector => SparkMLDenseVector, SparseVector => SparkMLSparseVector, Vector => SparkMLVector}
import org.apache.spark.mllib.linalg.{DenseVector => SparkDenseVector, SparseVector => SparkSparseVector, Vector => SparkVector}

/**
 * A utility object that contains operations to create, copy, compare, and convert [[Vector]] objects.
 */
object VectorUtils {

  protected[ml] val SPARSE_VECTOR_ACTIVE_SIZE_TO_SIZE_RATIO: Double = 1D / 3D
  protected[ml] val DEFAULT_SPARSITY_THRESHOLD: Double = 1e-4

  /**
   * Convert an [[Array]] of ([[Int]], [[Double]]) pairs into a [[Vector]].
   *
   * @param indexAndData An [[Array]] of ([[Int]], [[Double]]) pairs of indices and data to be converted to a [[Vector]]
   * @param length The length of the resulting vector
   * @param sparseVectorActiveSizeToSizeRatio The ratio used to determine whether a [[DenseVector]] or a
   *                                          [[SparseVector]] should be used to represent the underlying [[Vector]].
   *                                          For example, if the active size of the underlying vector is smaller than
   *                                          the length * sparseVectorActiveSizeToSizeRatio, then the [[SparseVector]]
   *                                          is chosen to represent the underlying [[Vector]], otherwise
   *                                          [[DenseVector]] is chosen.
   * @return The converted [[Vector]]
   */
  protected[ml] def toVector(
      indexAndData: Array[(Int, Double)],
      length: Int,
      sparseVectorActiveSizeToSizeRatio: Double = SPARSE_VECTOR_ACTIVE_SIZE_TO_SIZE_RATIO): Vector[Double] =
    if (length * SPARSE_VECTOR_ACTIVE_SIZE_TO_SIZE_RATIO < indexAndData.length) {
      toDenseVector(indexAndData, length)
    } else {
      toSparseVector(indexAndData, length)
    }

  /**
   * Convert an [[Array]] of ([[Int]], [[Double]]) pairs into a [[SparseVector]].
   *
   * @note Does not check for repeated indices.
   *
   * @param indexAndData An [[Array]] of ([[Int]], [[Double]]) pairs
   * @param length The length of the resulting sparse vector
   * @return The converted [[SparseVector]]
   */
  protected[ml] def toSparseVector(indexAndData: Array[(Int, Double)], length: Int): SparseVector[Double] = {

    // This would be all we need, but breeze doesn't like it.
    //SparseVector[Double](length)(indexAndData.sortBy(_._1):_*)
    val sortedIndexAndData = indexAndData.sortBy(_._1)
    val (index, data) = sortedIndexAndData.unzip

    new SparseVector[Double](index.toArray, data.toArray, length)
  }

  /**
   * Convert an [[Array]] of ([[Int]], [[Double]]) pairs into a [[DenseVector]].
   *
   * @note Does not check for repeated indices.
   *
   * @param indexAndData An [[Array]] of ([[Int]], [[Double]]) pairs
   * @param length The length of the resulting dense vector
   * @return The converted [[DenseVector]]
   */
  protected[ml] def toDenseVector(indexAndData: Array[(Int, Double)], length: Int): DenseVector[Double] = {

    val data = new Array[Double](length)

    indexAndData.foreach(i => data(i._1) = i._2)

    new DenseVector[Double](data)
  }

  /**
   * Initialize a zero vector of the same type as the input prototype vector (i.e. if the prototype vector is
   * a sparse vector, then the new vector should also be sparse).
   *
   * @param prototypeVector The prototype vector
   * @return A newly initialized zero vector
   */
  def zeroOfSameType(prototypeVector: Vector[Double]): Vector[Double] =

    prototypeVector match {
      case dense: DenseVector[Double] => DenseVector.zeros[Double](dense.length)
      case sparse: SparseVector[Double] => SparseVector.zeros[Double](sparse.length)
      case _ =>
        throw new IllegalArgumentException(
          s"Vector types other than ${classOf[DenseVector[Double]]} and ${classOf[SparseVector[Double]]} are not " +
            s"supported. Current class type: ${prototypeVector.getClass.getName}.")
    }

  /**
   * The Kronecker product between two vectors. Wiki reference on the Kronecker product (also referred to as the "outer
   * product"): [[https://en.wikipedia.org/wiki/Kronecker_product]].
   *
   * @param vector1 The left vector
   * @param vector2 The right vector
   * @param threshold Threshold of the cross value (any value below the threshold will be recorded as 0)
   * @return The resulting Kronecker product between vector1 and vector2
   */
  protected[ml] def kroneckerProduct(
      vector1: Vector[Double],
      vector2: Vector[Double],
      threshold: Double = DEFAULT_SPARSITY_THRESHOLD): Vector[Double] = {

    require(vector1.isInstanceOf[SparseVector[Double]] || vector2.isInstanceOf[SparseVector[Double]],
      "Kronecker product between two dense vectors is currently not supported")

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

  /**
   * Converts a Breeze vector to an spark.mllib vector.
   *
   * @note Lifted from spark private API.
   *
   * @param breezeVector The Breeze vector
   * @return The mllib vector
   */
  @deprecated("Photon ML is moving off of spark.mlllib.Vector to spark.ml.Vector", "")
  def breezeToMllib(breezeVector: Vector[Double]): SparkVector = {

    breezeVector match {
      case v: DenseVector[Double] =>
        if (v.offset == 0 && v.stride == 1 && v.length == v.data.length) {
          new SparkDenseVector(v.data)
        } else {
          new SparkDenseVector(v.toArray)  // Can't use underlying array directly, so make a new one
        }

      case v: SparseVector[Double] =>
        if (v.index.length == v.used) {
          new SparkSparseVector(v.length, v.index, v.data)
        } else {
          new SparkSparseVector(v.length, v.index.slice(0, v.used), v.data.slice(0, v.used))
        }

      case v: Vector[_] =>
        throw new IllegalArgumentException("Unsupported Breeze vector type: " + v.getClass.getName)
    }
  }

  /**
   * Converts a Breeze vector to a spark.ml vector.
   *
   * @note Lifted from spark private API.
   *
   * @param breezeVector The Breeze vector
   * @return The ml vector
   */
  def breezeToMl(breezeVector: Vector[Double]): SparkMLVector = {

    breezeVector match {
      case v: DenseVector[Double] =>
        if (v.offset == 0 && v.stride == 1 && v.length == v.data.length) {
          new SparkMLDenseVector(v.data)
        } else {
          new SparkMLDenseVector(v.toArray)  // Can't use underlying array directly, so make a new one
        }

      case v: SparseVector[Double] =>
        if (v.index.length == v.used) {
          new SparkMLSparseVector(v.length, v.index, v.data)
        } else {
          new SparkMLSparseVector(v.length, v.index.slice(0, v.used), v.data.slice(0, v.used))
        }

      case v: Vector[_] =>
        throw new IllegalArgumentException("Unsupported Breeze vector type: " + v.getClass.getName)
    }
  }

  /**
   * Converts a mllib vector to a Breeze vector.
   *
   * @note Lifted from spark private API.
   *
   * @param mllibVector The mllib vector
   * @return The Breeze vector
   */
  def mllibToBreeze(mllibVector: SparkVector): Vector[Double] =

    mllibVector match {
      case v: SparkSparseVector =>
        new SparseVector[Double](v.indices, v.values, v.size)

      case v: SparkDenseVector =>
        new DenseVector[Double](v.values)

      case v =>
        throw new IllegalArgumentException("Unsupported mllib vector type: " + v.getClass.getName)
    }

  /**
   * Converts a spark.ml vector to a Breeze vector.
   *
   * @todo This is just a wrapper for now, but at some point this class should be rewritten in terms of spark.ml Vector
   *
   * @param mlVector The spark.ml vector
   * @return The Breeze vector
   */
  def mlToBreeze(mlVector: SparkMLVector): Vector[Double] =
    mlVector match {
      case dv: SparkMLDenseVector =>
        new DenseVector[Double](dv.values)
      case sv: SparkMLSparseVector =>
        new SparseVector[Double](sv.indices, sv.values, sv.size)
    }

  /**
   * Determines when two vectors are "equal" within a very small tolerance.
   *
   * @note Zip stops without an error when the shortest argument stops! For that reason, we are going to return false if
   *       the 2 vectors have different lengths.
   *
   * @param v1 The first vector
   * @param v2 The second vector
   * @return True if the two vectors are "equal within epsilon", false otherwise
   */
  def areAlmostEqual(v1: Vector[Double], v2: Vector[Double]): Boolean =

    v1.length == v2.length && v1.toArray.zip(v2.toArray).forall { case (m1, m2) =>
      MathUtils.isAlmostZero(m2 - m1)
    }

  /**
   * Returns the indices for non-zero elements of the vector
   *
   * @param vector the input vector
   * @return the set of indices
   */
  def getActiveIndices(vector: Vector[Double]): mutable.Set[Int] = vector match {
    case dense: DenseVector[Double] =>
      val set: mutable.Set[Int] = mutable.Set.empty[Int]
      var index: Int = 0

      dense
        .valuesIterator
        .foreach { value =>
          if (!MathUtils.isAlmostZero(value)) {
            set += index
          }

          index += 1
        }

      set

    case sparse: SparseVector[Double] =>
      val set: mutable.Set[Int] = mutable.Set.empty[Int]

      sparse.activeKeysIterator.foreach(set += _)

      set
  }
}
