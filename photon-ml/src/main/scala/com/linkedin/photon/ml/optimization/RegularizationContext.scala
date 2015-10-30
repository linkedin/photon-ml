package com.linkedin.photon.ml.optimization

import RegularizationType.RegularizationType
import com.linkedin.photon.ml.function.{TwiceDiffFunction, DiffFunction}


/**
 * The regularization context holds the information of the regularization type (L1, L2, Elastic net) and the alpha value
 * if the regularization type is elastic net.
 *
 * The regularization term will be lambda [ (1-alpha)/2 ||w||,,2,,^2^ + alpha ||w||,,1,, ]
 *
 * The 1/2 factor for L2 regularization is handled in [[TwiceDiffFunction#withRegularization TwiceDiffFunction]]
 * and [[DiffFunction#withRegularization DiffFunction]] so the L2 regularization weight here does not have 1/2 factor.
 *
 * If alpha = 1, it is an L1. If alpha = 0, it is an L2.
 *
 * @author dpeng
 */
class RegularizationContext(val regularizationType: RegularizationType, elasticNetParam: Option[Double] = None) {
  val alpha: Double = (regularizationType, elasticNetParam) match {
    case (RegularizationType.ELASTIC_NET, Some(x)) if x > 0.0d && x <= 1.0d => x
    case (RegularizationType.L1, None) => 1.0d
    case (RegularizationType.L2, None) => 0.0d
    case _ => throw new IllegalArgumentException(s"Wrong input: RegularizationContext($regularizationType, $elasticNetParam)")
  }

  /**
   * Return the weight for the L1 regularization
   * @param lambda The regularization parameter
   * @return the coefficient for L1 regularization
   */
  def getL1RegularizationWeight(lambda: Double): Double = alpha * lambda

  /**
   * Return the weight for the L2 regularization
   * @param lambda The regularization parameter
   * @return the coefficient for L2 regularization
   */
  def getL2RegularizationWeight(lambda: Double): Double = (1 - alpha) * lambda
}

/**
 * A singleton object for L1 regularization
 */
object L1RegularizationContext extends RegularizationContext(RegularizationType.L1)

/**
 * A singleton object for L2 regularization
 */
object L2RegularizationContext extends RegularizationContext(RegularizationType.L2)
