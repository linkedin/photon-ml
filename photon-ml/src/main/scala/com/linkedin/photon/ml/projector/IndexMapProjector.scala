package com.linkedin.photon.ml.projector

import scala.collection.Map

import breeze.linalg.{SparseVector, Vector}

import com.linkedin.photon.ml.util.VectorUtils

/**
 * A projection map that maintains the one-to-one mapping of indices between the original and projected space
 *
 * @param originalToProjectedSpaceMap map from original to projected space
 * @param originalSpaceDimension dimensionality of the original space
 * @param projectedSpaceDimension dimensionality of the projected space
 * @author xazhang
 */
class IndexMapProjector(
    originalToProjectedSpaceMap: Map[Int, Int],
    override val originalSpaceDimension: Int,
    override val projectedSpaceDimension: Int) extends Projector {

  private val projectedToOriginalSpaceMap = originalToProjectedSpaceMap.map(_.swap)

  assert(originalToProjectedSpaceMap.size == projectedToOriginalSpaceMap.size, s"The size of " +
      s"originalToProjectedSpaceMap (${originalToProjectedSpaceMap.size}) and the size of " +
      s"projectedToOriginalSpaceMap (${projectedToOriginalSpaceMap.size}) are expected to be equal, " +
      s"e.g., there should exist a one-to-one correspondence between the indices in the original and projected space.")

  /**
   * Project features into the new space
   *
   * @param features the features
   * @return projected features
   */
  override def projectFeatures(features: Vector[Double]): Vector[Double] = {
    IndexMapProjector.projectWithMap(features, originalToProjectedSpaceMap, projectedSpaceDimension)
  }

  /**
   * Project coefficients into the new space
   *
   * @param coefficients the coefficients
   * @return projected coefficients
   */
  override def projectCoefficients(coefficients: Vector[Double]): Vector[Double] = {
    IndexMapProjector.projectWithMap(coefficients, projectedToOriginalSpaceMap, originalSpaceDimension)
  }
}

object IndexMapProjector {

  /**
   * Generate the index map projector given an iterator of feature vectors
   *
   * @param features An [[Iterable]] of feature vectors
   * @return The generated projection map
   */
  def buildIndexMapProjector(features: Iterable[Vector[Double]]): IndexMapProjector = {
    val originalToProjectedSpaceMap = features.flatMap(_.activeKeysIterator).toSet[Int].zipWithIndex.toMap
    val originalSpaceDimension = features.head.length
    val projectedSpaceDimension = originalToProjectedSpaceMap.values.max + 1

    new IndexMapProjector(originalToProjectedSpaceMap, originalSpaceDimension, projectedSpaceDimension)
  }

  /**
   * Project the indices of the input vector with the given map
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

    VectorUtils.indexAndValueArrayToVector(indexAndData, dimension)
  }
}
