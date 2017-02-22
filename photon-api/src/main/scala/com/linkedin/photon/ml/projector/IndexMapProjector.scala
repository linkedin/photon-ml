/*
 * Copyright 2016 LinkedIn Corp. All rights reserved.
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

import scala.collection.Map

import breeze.linalg.Vector
import com.linkedin.photon.ml.util.VectorUtils

/**
 * A projection map that maintains the one-to-one mapping of indices between the original and projected space.
 *
 * This is expected to be used in cases where we are training models on a subset of features so that it would not make
 * sense to deal with vectors of the size of the original dimension.
 *
 * e.g. If one is training a per-item model, the instances in the training set that is used for this task might only
 *      have a small subset of the features active. In such a case, we can train a model in the smaller space for
 *      better efficiency but we need to also map the coefficients back to the original space eventually. This class
 *      does the projection of features to the smaller sub-space as well as projecting the features back to the
 *      original dimensions
 *
 * Given the intended use case, no primary constructor is provided. It can only be constructed via the companion
 * object's builder method.
 *
 * @param originalToProjectedSpaceMap Map from original to projected space
 * @param originalSpaceDimension Dimensionality of the original space
 * @param projectedSpaceDimension Dimensionality of the projected space
 */
protected[ml] class IndexMapProjector private (
    val originalToProjectedSpaceMap: Map[Int, Int],
    override val originalSpaceDimension: Int,
    override val projectedSpaceDimension: Int) extends Projector {

  val projectedToOriginalSpaceMap = originalToProjectedSpaceMap.map(_.swap)

  assert(originalToProjectedSpaceMap.size == projectedToOriginalSpaceMap.size, s"The size of " +
      s"originalToProjectedSpaceMap (${originalToProjectedSpaceMap.size}) and the size of " +
      s"projectedToOriginalSpaceMap (${projectedToOriginalSpaceMap.size}) are expected to be equal, " +
      s"e.g., there should exist a one-to-one correspondence between the indices in the original and projected space.")

  /**
   * Project features into the new space.
   *
   * @param features The features
   * @return Projected features
   */
  override def projectFeatures(features: Vector[Double]): Vector[Double] = {
    IndexMapProjector.projectWithMap(features, originalToProjectedSpaceMap, projectedSpaceDimension)
  }

  /**
   * Project coefficients into the new space.
   *
   * @param coefficients The coefficients
   * @return Projected coefficients
   */
  override def projectCoefficients(coefficients: Vector[Double]): Vector[Double] = {
    IndexMapProjector.projectWithMap(coefficients, projectedToOriginalSpaceMap, originalSpaceDimension)
  }
}

object IndexMapProjector {

  /**
   * Generate the index map projector given an iterator of feature vectors.
   *
   * @param features An [[Iterable]] of feature vectors
   * @return The generated projection map
   */
  protected[ml] def buildIndexMapProjector(features: Iterable[Vector[Double]]): IndexMapProjector = {
    val originalToProjectedSpaceMap = features.flatMap(_.activeKeysIterator).toSet[Int].zipWithIndex.toMap
    val originalSpaceDimension = features.head.length
    val projectedSpaceDimension = originalToProjectedSpaceMap.values.max + 1

    new IndexMapProjector(originalToProjectedSpaceMap, originalSpaceDimension, projectedSpaceDimension)
  }

  /**
   * Project the indices of the input vector with the given map.
   *
   * @param vector The input vector in the original space
   * @param map The projection map
   * @param dimension The dimension of the projected space
   * @return The output vector in the projected space
   */
  private def projectWithMap(vector: Vector[Double], map: Map[Int, Int], dimension: Int): Vector[Double] = {
    val indexAndData = vector.activeIterator
      .filter { case (key, _) => map.contains(key) }
      .map { case (key, value) => (map(key), value) }.toArray

    VectorUtils.convertIndexAndValuePairArrayToVector(indexAndData, dimension)
  }
}
