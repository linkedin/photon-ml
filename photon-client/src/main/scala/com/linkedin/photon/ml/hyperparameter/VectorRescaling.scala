/*
 * Copyright 2020 LinkedIn Corp. All rights reserved.
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
package com.linkedin.photon.ml.hyperparameter

import breeze.linalg.DenseVector

import com.linkedin.photon.ml.util.DoubleRange

/**
 * Functions to transform/scale forward/backward hyper-parameters in vectors
 */
object VectorRescaling {

  val LOG_TRANSFORM = "LOG"
  val SQRT_TRANSFORM = "SQRT"

  /**
   * Apply forward scaling to a vector. Given a range [a, b] and an element x of the vector,
   * y = (x - a) / (b - a), if x is continuous;
   * y = (x - a) / (b - a + 1), if x is discrete.
   *
   * @param vector A DenseVector.
   * @param ranges A sequence of ranges for every element in the vector to be scaled.
   * @param discreteIndexSet A Set with indices of discrete elements in the vector.
   * @return The scaled vector.
   */
  def scaleForward(
      vector: DenseVector[Double],
      ranges: Seq[DoubleRange],
      discreteIndexSet: Set[Int] = Set()): DenseVector[Double] = {

    val vectorScaled = vector.copy

    val start = DenseVector(ranges.map(_.start).toArray)
    val end = DenseVector(ranges.map(_.end).toArray)
    val discreteAdj = DenseVector.tabulate(ranges.length) { i =>
      if (discreteIndexSet.contains(i)) 1.0 else 0.0
    }

    vectorScaled :-= start
    vectorScaled :/= (end - start + discreteAdj)

    vectorScaled
  }

  /**
   * Apply backward scaling to a vector. Given range [a, b] and an element x of the vector,
   * y = x * (b - a) + a, if x is continuous;
   * y = x * (b - a + 1) + a, if x is discrete.
   *
   * @param vector A DenseVector.
   * @param ranges A sequence of ranges for every element in the vector to be scaled.
   * @param discreteIndexSet A Set with indices of discrete elements in the vector.
   * @return The scaled vector.
   */
  def scaleBackward(
      vector: DenseVector[Double],
      ranges: Seq[DoubleRange],
      discreteIndexSet: Set[Int] = Set()): DenseVector[Double] = {

    val vectorScaled = vector.copy

    val start = DenseVector(ranges.map(_.start).toArray)
    val end = DenseVector(ranges.map(_.end).toArray)
    val discreteAdj = DenseVector.tabulate(ranges.length) { i =>
      if (discreteIndexSet.contains(i)) 1.0 else 0.0
    }

    vectorScaled :*= (end - start + discreteAdj)
    vectorScaled :+= start

    vectorScaled
  }
}
