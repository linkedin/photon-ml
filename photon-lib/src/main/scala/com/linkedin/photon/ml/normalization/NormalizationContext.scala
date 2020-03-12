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
package com.linkedin.photon.ml.normalization

import breeze.linalg.{DenseVector, Vector}

import com.linkedin.photon.ml.normalization.NormalizationType.NormalizationType
import com.linkedin.photon.ml.stat.FeatureDataStatistics

/**
 * The transformation consists of up to two parts: a translational shift and a scaling factor. The normalization of a
 * feature vector x is:
 *
 * x' = (x .- shift) .* factor
 *
 * where the operations are point-wise. A missing shift vector is equivalent to the 0 vector (all 0s), and a missing
 * factor vector is equivalent to the 1 vector (all 1s).
 *
 * If the shift is enabled, there must be an intercept provided. This class assume that the intercepts for the original
 * and the transformed space are both 1, so the shift for the intercept should be 0, and the factor for the intercept
 * should be 1.
 *
 * Note that this class covers all affine transformations, excluding rotation.
 */
protected[ml] class NormalizationContext(
    val factorsOpt: Option[Vector[Double]],
    val shiftsAndInterceptOpt: Option[(Vector[Double], Int)])
  extends Serializable {

  val size: Int = (factorsOpt, shiftsAndInterceptOpt) match {
    case (Some(factors), None) =>
      factors.length

    case (None, Some((shifts, intercept))) =>
      require(shifts(intercept) == 0D, s"The intercept should not be shifted: ${shifts(intercept)} != 0")

      shifts.length

    case (Some(factors), Some((shifts, intercept))) =>
      require(shifts(intercept) == 0D, s"The intercept should not be shifted: ${shifts(intercept)} != 0")
      require(factors(intercept) == 1D, s"The intercept should not be scaled: ${factors(intercept)} != 1")
      require(factors.size == shifts.size, "Factors and shifts vectors should have the same size")

      factors.length

    case (None, None) =>
      0
  }

  /**
   * Transform the model coefficients of the transformed space to the original space. The key requirement for the
   * transformation is to keep the margin consistent in both spaces, i.e:
   *
   * w^T^ x + b = w'^T^ x' + b' = w'^T^ [(x - shift) .* factor] + b'
   *
   * where b is the explicit intercept, and .* is a point wise multiplication. To make the equation work for all x, we
   * have:
   *
   * w = w' .* factor
   * b = - w ^T^ shift + b'
   *
   * @param inputCoef The coefficients + the intercept (if present) in the transformed space
   * @return The coefficients + the intercept (if present) in the original space
   */
  def coefToOriginalSpace(inputCoef: Vector[Double]): Vector[Double] =
    if (size == 0) {
      inputCoef
    } else {
      require(size == inputCoef.size, "Vector size and the scaling factor/shift size are different.")

      val outputCoef = inputCoef.copy

      factorsOpt.foreach { factors =>
        outputCoef :*= factors
      }
      // All shifts go to intercept
      shiftsAndInterceptOpt.foreach { case (shifts, intercept) =>
        outputCoef(intercept) -= outputCoef.dot(shifts)
      }

      outputCoef
    }

  /**
   * Transform the model coefficients of the original space to the transformed space. The key requirement for the
   * transformation is to keep the margin consistent in both spaces, i.e:
   *
   * w^T^ x + b = w'^T^ x' + b' = w'^T^ [(x - shift) .* factor] + b'
   *
   * where b is the explicit intercept, and .* is a point wise multiplication. To make the equation work for all x, we
   * have:
   *
   * w' = w ./ factor
   * b' = w^T^ shift + b
   *
   * @param inputCoef The coefficients + the intercept (if present) in the original space
   * @return The coefficients + the intercept (if present) in the transformed space
   */
  def coefToTransformedSpace(inputCoef: Vector[Double]): Vector[Double] =
    if (size == 0) {
      inputCoef
    } else {
      require(size == inputCoef.size, "Vector size and the scaling factor/shift size are different.")

      val outputCoef = inputCoef.copy

      // All shifts go to intercept
      shiftsAndInterceptOpt.foreach { case (shifts, intercept) =>
        outputCoef(intercept) += outputCoef.dot(shifts)
      }
      factorsOpt.foreach { factors =>
        outputCoef :/= factors
      }

      outputCoef
    }

  /**
   * Transform the variance of the model coefficients of the original space to the transformed space. Will be used in
   * constructing the prior.
   *
   * w' = w ./ factor
   * b' = w^T^ shift + b
   *
   * Then variance transformation formula is
   *
   * Var(w') = Var(w) ./ (factor .* factor)
   * Var(b') = Var(b)
   *
   * @param inputVar The variance of coefficients + the intercept (if present) in the transformed space
   * @return The variance of coefficients + the intercept (if present) in the original space
   */
  def varToTransformedSpace(inputVar: Vector[Double]): Vector[Double] =
    if (size == 0) {
      inputVar
    } else {
      require(size == inputVar.size, "Vector size and the scaling factor size are different.")

      val outputVar = inputVar.copy

      factorsOpt.foreach { factors =>
        // Factors of intercept is 1
        outputVar :/= (factors *:* factors)
      }
      outputVar
    }
}

protected[ml] object NormalizationContext {

  /**
   * A factory method to create a normalization context according to the [[NormalizationType]] and the
   * feature summary. If using [[NormalizationType.STANDARDIZATION]], an intercept index is also needed.
   *
   * @param normalizationType The normalization type
   * @param summary Features statistical summary
   * @return A normalization context
   */
  def apply(
      normalizationType: NormalizationType,
      summary: FeatureDataStatistics): NormalizationContext = normalizationType match {

    case NormalizationType.NONE =>
      NoNormalization()

    case NormalizationType.SCALE_WITH_MAX_MAGNITUDE =>
      val factors = summary
        .max
        .toArray
        .zip(summary.min.toArray)
        .map { case (max, min) =>
          val magnitude = math.max(math.abs(max), math.abs(min))
          if (magnitude == 0) 1.0 else 1.0 / magnitude
        }

      new NormalizationContext(Some(DenseVector(factors)), None)

    case NormalizationType.SCALE_WITH_STANDARD_DEVIATION =>
      val factors = summary
        .variance
        .map { s =>
          val std = math.sqrt(s)

          if (std == 0) 1.0 else 1.0 / std
        }

      new NormalizationContext(Some(factors), None)

    case NormalizationType.STANDARDIZATION =>
      val factors = summary
        .variance
        .map { s =>
          val std = math.sqrt(s)

          if (std == 0) 1.0 else 1.0 / std
        }
      val shifts = summary.mean.copy
      val interceptId = summary.interceptIndex.get

      // Do not transform intercept
      shifts(interceptId) = 0.0
      factors(interceptId) = 1.0

      new NormalizationContext(Some(factors), Some((shifts, interceptId)))

    case _ =>
      throw new IllegalArgumentException(s"NormalizationType $normalizationType not recognized.")
  }

  /**
   * Convenience method to extract construction arguments.
   *
   * @param context An existing [[NormalizationContext]]
   * @return The contstruction arguments of the [[NormalizationContext]]
   */
  def unapply(context: NormalizationContext): Option[(Option[Vector[Double]], Option[(Vector[Double], Int)])] =
    if (context == null) {
      None
    } else {
      Some(context.factorsOpt, context.shiftsAndInterceptOpt)
    }
}

/**
 * Factory to create contexts for no normalization.
 */
protected[ml] object NoNormalization {

  private val none: NormalizationContext = new NormalizationContext(None, None)

  /**
   * Constructor ex nihilo, comme appelé du néant.
   *
   * @return An instance of NoNormalizationContext
   */
  def apply(): NormalizationContext = none
}
