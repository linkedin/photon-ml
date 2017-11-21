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

import breeze.linalg.{DenseVector, Vector}
import breeze.optimize.{FirstOrderMinimizer, DiffFunction => BreezeDiffFunction, LBFGSB => BreezeLBFGSB}
import org.apache.spark.broadcast.Broadcast

import com.linkedin.photon.ml.function.DiffFunction
import com.linkedin.photon.ml.normalization.NormalizationContext

/**
 * Class used to solve an optimization problem using Limited-memory BFGS-B (LBFGSB) with simple box constrains.
 * Reference: [[http://epubs.siam.org/doi/10.1137/0916069]].
 *
 * @param lowerBounds The lower bounds for feature coefficients
 * @param upperBounds The upper bounds for feature coefficients
 * @param normalizationContext The normalization context
 * @param tolerance The tolerance threshold for improvement between iterations as a percentage of the initial loss
 * @param maxNumIterations The cut-off for number of optimization iterations to perform.
 * @param numCorrections The number of corrections used in the LBFGS update. Default 10. Values of numCorrections less
 *                       than 3 are not recommended; large values of numCorrections will result in excessive computing
 *                       time.
 *                       Recommended:  3 < numCorrections < 10
 *                       Restriction:  numCorrections > 0
 * @param isTrackingState Whether to track intermediate states during optimization
 */
class LBFGSB (
    lowerBounds: DenseVector[Double],
    upperBounds: DenseVector[Double],
    normalizationContext: Broadcast[NormalizationContext],
    numCorrections: Int = LBFGS.DEFAULT_NUM_CORRECTIONS,
    tolerance: Double = LBFGS.DEFAULT_TOLERANCE,
    maxNumIterations: Int = LBFGS.DEFAULT_MAX_ITER,
    isTrackingState: Boolean = Optimizer.DEFAULT_TRACKING_STATE)
  extends LBFGS(
    normalizationContext,
    numCorrections,
    tolerance,
    maxNumIterations,
    isTrackingState) {

  /**
   * Under the hood, this adaptor uses an LBFGSB optimizer from Breeze to optimize functions.
   * The DiffFunction is modified into a Breeze DiffFunction which the Breeze optimizer can understand.
   */
  protected class BreezeBOptimization(
      breezeDiffFunction: BreezeDiffFunction[DenseVector[Double]],
      initCoefficients: Vector[Double])
    extends BreezeOptimization(breezeDiffFunction.asInstanceOf[BreezeDiffFunction[Vector[Double]]], initCoefficients)

  // Cast into FirstOrderMinimizer as for overriding breezeOptimizer in parent class LBFGS
  override protected val breezeOptimizer = new BreezeLBFGSB(lowerBounds, upperBounds, maxNumIterations, numCorrections, tolerance).
    asInstanceOf[FirstOrderMinimizer[Vector[Double], BreezeDiffFunction[Vector[Double]]]]

  /**
   * Initialize breeze optimization engine.
   *
   * @param objectiveFunction The objective function to be optimized
   * @param initState         The initial state of the optimizer, prior to starting optimization
   * @param data              The training data
   */
  override def init(objectiveFunction: DiffFunction, initState: OptimizerState)(data: objectiveFunction.Data): Unit = {
    val breezeDiffFunction: BreezeDiffFunction[DenseVector[Double]] = new BreezeDiffFunction[DenseVector[Double]] {
      override def calculate(x: DenseVector[Double]): (Double, DenseVector[Double]) = {
        var convertedCoefficients: objectiveFunction.Coefficients = objectiveFunction.convertFromVector(x)
        var result = objectiveFunction.calculate(data, convertedCoefficients, normalizationContext)
        (result._1, result._2.toDenseVector)
      }
    }
    breezeOptimization = new BreezeBOptimization(breezeDiffFunction, initState.coefficients)
  }

  /**
   * Just reset the whole BreezeOptimization instance.
   */
  override def clearOptimizerInnerState(): Unit = {

    super.clearOptimizerInnerState()
    breezeOptimization = _: BreezeBOptimization
  }
}
