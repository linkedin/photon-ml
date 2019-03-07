/*
 * Copyright 2019 LinkedIn Corp. All rights reserved.
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
package com.linkedin.photon.ml.projector

import breeze.linalg.Vector

import com.linkedin.photon.ml.util.VectorUtils

/**
 * Project [[Vector]] objects between spaces, where the projected space is a linear subspace of the original space.
 *
 * An example use case is training models on a subset of features, where a reduction in vector size will greatly
 * improve performance.
 *
 *    e.g. When training a per-entity (per-user, per-item, etc.) model, the entity samples may contain only a small
 *         subset of the active features across all entities. In such a case, a reduction of the sample vector dimension
 *         to only the active features results in better performance. This reduction can be trivially reversed after
 *         model training.
 *
 * @param subspaceIndices The set of indices corresponding to the axes of the original space to preserve and use to form
 *                        the subspace.
 * @param dimension The dimension of the original space
 */
protected[ml] class LinearSubspaceProjector(subspaceIndices: Set[Int], dimension: Int) extends Serializable {

  import LinearSubspaceProjector._

  val originalSpaceDimension: Int = dimension
  val projectedSpaceDimension: Int = subspaceIndices.size
  val originalToProjectedSpaceMap: Map[Int, Int] = subspaceIndices.zipWithIndex.toMap
  val projectedToOriginalSpaceMap: Map[Int, Int] = originalToProjectedSpaceMap.map(_.swap)

  require(
    subspaceIndices.forall(_ < dimension),
    "Given dimension of original space less than one or more numbered indices in set of subspace indices.")

  /**
   * Project [[Vector]] to subspace.
   *
   * @param input A [[Vector]] in the original space
   * @return The same [[Vector]] in the projected space
   */
  def projectForward(input: Vector[Double]): Vector[Double] =
    remapVector(input, originalToProjectedSpaceMap, projectedSpaceDimension)

  /**
   * Project coefficients into the new space.
   *
   * @param input A [[Vector]] in the projected space
   * @return The same [[Vector]] in the original space
   */
  def projectBackward(input: Vector[Double]): Vector[Double] =
    remapVector(input, projectedToOriginalSpaceMap, originalSpaceDimension)
}

object LinearSubspaceProjector {

  /**
   * Create a new [[Vector]] by mapping the indices of an existing [[Vector]].
   *
   * @param vector The input [[Vector]]
   * @param map The map of old index to new index
   * @param dimension The dimension of the new [[Vector]]
   * @return A new [[Vector]] with re-mapped indices
   */
  private def remapVector(vector: Vector[Double], map: Map[Int, Int], dimension: Int): Vector[Double] = {

    val indexAndData = vector
      .activeIterator
      .filter { case (key, _) => map.contains(key) }
      .map { case (key, value) => (map(key), value) }
      .toArray

    VectorUtils.toVector(indexAndData, dimension)
  }
}
