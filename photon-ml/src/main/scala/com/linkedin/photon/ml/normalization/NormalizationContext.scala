package com.linkedin.photon.ml.normalization

import breeze.linalg.{DenseVector, Vector}
import com.linkedin.photon.ml.stat.BasicStatisticalSummary


/**
 * Intercept as a feature value is always one.
 * @author dpeng
 */
case class NormalizationContext(factors: Option[_ <: Vector[Double]], shifts: Option[_ <: Vector[Double]],
                                interceptId: Option[Int]) {
  require(!(shifts.isDefined && interceptId.isEmpty), "Shift without intercept is illegal.")
  if (factors.isDefined && shifts.isDefined) require(factors.get.size == shifts.get.size,
                                                     "Factors and shifts vectors should have the same size")

  /**
   * Transform the coefficients to original form
   * @param inputCoef
   * @return
   */
  def transformModelCoefficients(inputCoef: Vector[Double]): Vector[Double] = {
    val outputCoef = factors match {
      case Some(fs) =>
        inputCoef :* fs
      case None =>
        inputCoef.copy
    }
    // All shifts go to intercept
    shifts.foreach(ss => {
      outputCoef(interceptId.get) -= outputCoef.dot(ss)
    })
    outputCoef
  }

  /**
   * For testing purpose only. This is not designed to be efficient. Transform the vector to the normalized form
   * @param input
   * @return
   */
  def transformVector(input: Vector[Double]): Vector[Double] = {
    (factors, shifts) match {
      case (Some(fs), Some(ss)) =>
        require(fs.size == input.size, "Vector size and the scaling factor size are different.")
        (input - ss) :* fs
      case (Some(fs), None) =>
        require(fs.size == input.size, "Vector size and the scaling factor size are different.")
        input :* fs
      case (None, Some(ss)) =>
        require(ss.size == input.size, "Vector size and the scaling factor size are different.")
        input - ss
      case (None, None) =>
        input
    }
  }
}

object NormalizationContext {
  def apply(normalizationType: NormalizationType, summary: => BasicStatisticalSummary,
      interceptId: Option[Int]): NormalizationContext = {
    normalizationType match {
      case NormalizationType.NONE =>
        new NormalizationContext(None, None, interceptId)
      case NormalizationType.SCALE_WITH_MAX_MAGNITUDE =>
        val factors = summary.max.toArray.zip(summary.min.toArray).map {
          case (max, min) =>
            val magnitude = math.max(math.abs(max), math.abs(min))
            if (magnitude == 0) 1.0 else 1.0 / magnitude
          }
        new NormalizationContext(Some(DenseVector(factors)), None, interceptId)
      case NormalizationType.SCALE_WITH_STANDARD_DEVIATION =>
        val factors = summary.variance.map(x => {
          val std = math.sqrt(x)
          if (std == 0) 1.0 else 1.0 / std
        })
        new NormalizationContext(Some(factors), None, interceptId)
      case NormalizationType.STANDARDIZATION =>
        val factors = summary.variance.map(x => {
          val std = math.sqrt(x)
          if (std == 0) 1.0 else 1.0 / std
        })
        val shifts = summary.mean.copy
        // Do not transform intercept
        interceptId.foreach(id => {
          shifts(id) = 0.0
          factors(id) = 1.0
        })
        new NormalizationContext(Some(factors), Some(shifts), interceptId)
      case _ =>
        throw new IllegalArgumentException(s"NormalizationType $normalizationType not recognized.")
    }
  }
}

object NoNormalization extends NormalizationContext(factors = None, shifts = None, interceptId = None)
