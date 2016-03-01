package com.linkedin.photon.ml.projector

import breeze.linalg.Vector


/**
 * A trait that performs two types of projections:
 * <ul>
 * <li>
 *   Project the feature vector from the original space to the projected space, usually during model training phase.
 * </li>
 * <li>
 *   Project the coefficients from the projected space back to the original space, usually after model training and
 *   during the model storing nad postprocessing phase.
 * </li>
 * </ul>
 * @author xazhang
 */
protected[ml] trait Projector {

  /**
   * Dimension of the original space
   */
  val originalSpaceDimension: Int

  /**
   * Dimension of the projected space
   */
  val projectedSpaceDimension: Int

  /**
   * Project the feature vector from the original space to the projected space
   * @param features The input feature vector in the original space
   * @return The feature vector in the projected space
   */
  def projectFeatures(features: Vector[Double]): Vector[Double]

  /**
   * Project the coefficient vector from the projected space back to the original space
   * @param coefficients The input coefficient vector in the projected space
   * @return The coefficient vector in the original space
   */
  def projectCoefficients(coefficients: Vector[Double]): Vector[Double]
}
