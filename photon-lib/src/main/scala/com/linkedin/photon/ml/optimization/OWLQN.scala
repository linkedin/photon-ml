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
package com.linkedin.photon.ml.optimization

import breeze.linalg.Vector
import breeze.optimize.{OWLQN => BreezeOWLQN}

import com.linkedin.photon.ml.normalization.NormalizationContext
import com.linkedin.photon.ml.util.BroadcastWrapper

/**
 * L1 regularization has to be done differently because the gradient of L1 penalty term may be discontinuous.
 * For optimization with L1 penalty term, the optimization algorithm is a modified Quasi-Newton algorithm called OWL-QN.
 * Reference: [[http://research.microsoft.com/en-us/downloads/b1eb1016-1738-4bd5-83a9-370c9d498a03/]]
 *
 * @param l1RegWeight The L1 regularization weight
 * @param normalizationContext The normalization context
 * @param numCorrections The number of corrections used in the LBFGS update. Default 10. Values of numCorrections less
 *                       than 3 are not recommended; large values of numCorrections will result in excessive computing
 *                       time.
 *                       Recommended:  3 < numCorrections < 10
 *                       Restriction:  numCorrections > 0
 * @param tolerance The tolerance threshold for improvement between iterations as a percentage of the initial loss
 * @param maxNumIterations The cut-off for number of optimization iterations to perform.
 * @param constraintMap (Optional) The map of constraints on the feature coefficients
 */
class OWLQN(
    l1RegWeight: Double,
    normalizationContext: BroadcastWrapper[NormalizationContext],
    numCorrections: Int = LBFGS.DEFAULT_NUM_CORRECTIONS,
    tolerance: Double = LBFGS.DEFAULT_TOLERANCE,
    maxNumIterations: Int = LBFGS.DEFAULT_MAX_ITER,
    constraintMap: Option[Map[Int, (Double, Double)]] = Optimizer.DEFAULT_CONSTRAINT_MAP)
  extends LBFGS(
    normalizationContext,
    numCorrections,
    tolerance,
    maxNumIterations,
    constraintMap) {

  protected var regularizationWeight: Double = l1RegWeight

  /**
   * L1 regularization getter.
   *
   * @return The L1 regularization weight
   */
  def l1RegularizationWeight: Double = regularizationWeight

  /**
   * L1 regularization setter.
   *
   * @param newRegWeight The new L1 regularization weight
   */
  protected[optimization] def l1RegularizationWeight_=(newRegWeight: Double): Unit = {
    regularizationWeight = newRegWeight
  }

  /**
   * Under the hood, this adaptor uses an OWLQN
   * ([[http://www.scalanlp.org/api/breeze/index.html#breeze.optimize.OWLQN breeze.optimize.OWLQN]]) optimizer from
   * Breeze to optimize functions with L1 penalty term. The DiffFunction is modified into a Breeze DiffFunction which
   * the Breeze optimizer can understand. The L1 penalty is implemented in the optimizer level. See
   * [[http://www.scalanlp.org/api/breeze/index.html#breeze.optimize.OWLQN breeze.optimize.OWLQN]].
   */
  override protected val breezeOptimizer = new BreezeOWLQN[Int, Vector[Double]](
    maxNumIterations,
    numCorrections,
    (_: Int) => regularizationWeight,
    tolerance)
}
