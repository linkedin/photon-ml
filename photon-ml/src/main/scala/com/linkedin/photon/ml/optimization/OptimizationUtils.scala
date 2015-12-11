package com.linkedin.photon.ml.optimization

import breeze.linalg.Vector
import com.linkedin.photon.ml.util.Utils


/**
 * General purpose utils that can be leveraged by the various optimizers
 * @author nkatariy
 */
object OptimizationUtils {
  /**
   * Return new value of coeff after applying the bound constraint specified by the input tuple of lower and upper bound
   * @param coefficient value to be projected
   * @param bounds projection space
   * @return value of coeff after projection into constraint space. (bounds.get._1 <= newCoeffValue <= bounds.get._2)
   */
  private[this] def projectCoefficientToInterval(coefficient: Double, bounds: Option[(Double, Double)]): Double = {
    bounds match {
      case Some(x: (Double, Double)) =>
      val (lowerBound, upperBound) = x
        if (coefficient < lowerBound) {
          lowerBound
        } else if (coefficient > upperBound) {
          upperBound
        } else {
          coefficient
        }
      case None => coefficient
    }
  }

  /**
   * Return new coefficients after projection into the constrained space as specified by the map of feature index to
   * the (lowerBound, upperBound) constraints.
   * @param coefficients coefficients to be projected
   * @param constraintMap map of feature index to the bounds that are to be enforced on that particular feature
   * @return projected value of coefficients
   */
  def projectCoefficientsToHypercube(coefficients: Vector[Double], constraintMap: Option[Map[Int, (Double, Double)]]): Vector[Double] = {
    constraintMap match {
      case Some(x: Map[Int, (Double, Double)]) =>
        val projectedCoefficients = Utils.initializeZerosVectorOfSameType(coefficients)
        coefficients.activeIterator.foreach {
          case (key, value) =>
            if (x.contains(key)) {
              projectedCoefficients.update(key, projectCoefficientToInterval(value, x.get(key)))
            } else {
              projectedCoefficients.update(key, value)
            }
        }
        projectedCoefficients
      case None => coefficients
    }
  }
}