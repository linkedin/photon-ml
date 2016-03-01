package com.linkedin.photon.ml.model


import breeze.linalg.{Vector, norm}
import breeze.stats.meanAndVariance


/**
 * @param means The mean of the model coefficients
 * @param variancesOption The option of variance of the model coefficients
 * @author xazhang
 */
case class Coefficients(means: Vector[Double], variancesOption: Option[Vector[Double]]) {

  lazy val meansL2Norm: Double = norm(means, 2)
  lazy val variancesL2NormOption: Option[Double] = variancesOption.map(variances => norm(variances, 2))

  def computeScore(features: Vector[Double]): Double = {
    assert(means.length == features.length,
      s"Coefficients length (${means.length}}) != features length (${features.length}})")
    means.dot(features)
  }

  def toSummaryString: String = {
    val stringBuilder = new StringBuilder()
    stringBuilder.append(s"meanAndVarianceAndCount of the mean: ${meanAndVariance(means)}")
    stringBuilder.append(s"\nl2 norm of the mean: ${norm(means, 2)}")
    variancesOption.foreach(variances => s"\nmeanAndVarianceAndCount of variance: ${meanAndVariance(variances)}")
    stringBuilder.toString()
  }
}

object Coefficients {
  def initializeZeroCoefficients(dimension: Int): Coefficients = {
    Coefficients(Vector.zeros[Double](dimension), variancesOption = None)
  }
}
