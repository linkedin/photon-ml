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
package com.linkedin.photon.ml.optimization

import breeze.linalg.Vector
import breeze.optimize.{DiffFunction => BreezeDiffFunction, LBFGS => BreezeLBFGS, OWLQN => BreezeOWLQN}
import com.linkedin.photon.ml.data.DataPoint
import com.linkedin.photon.ml.function.{DiffFunction, L1RegularizationTerm}
import org.apache.spark.rdd.RDD

/**
 * Class used to solve an optimization problem using Limited-memory BFGS (LBFGS).
 * Reference: [[http://en.wikipedia.org/wiki/Limited-memory_BFGS]].
 * L1 regularization has to be done differently because the gradient of L1 penalty term may be discontinuous.
 * For optimization with L1 penalty term, the optimization algorithm is a modified Quasi-Newton algorithm called OWL-QN.
 * Reference: [[http://research.microsoft.com/en-us/downloads/b1eb1016-1738-4bd5-83a9-370c9d498a03/]]
 *
 * @param numCorrections
 * The number of corrections used in the LBFGS update. Default 10.
 * Values of numCorrections less than 3 are not recommended; large values
 * of numCorrections will result in excessive computing time.
 * 3 < numCorrections < 10 is recommended.
 * Restriction: numCorrections > 0
 * @tparam Datum Generic type of input data point
 * @author xazhang
 * @author dpeng
 * @author bdrew
 */
class LBFGS[Datum <: DataPoint](
    var numCorrections: Int = LBFGS.DEFAULT_NUM_CORRECTIONS)
  extends AbstractOptimizer[Datum, DiffFunction[Datum]](
    maxNumIterations = LBFGS.DEFAULT_MAX_ITER, tolerance = LBFGS.DEFAULT_TOLERANCE) {

  /**
   * Under the hood, this adaptor uses an LBFGS
   * ([[http://www.scalanlp.org/api/breeze/index.html#breeze.optimize.LBFGS breeze.optimize.LBFGS]]) optimizer from
   * Breeze to optimize functions without L1 penalty term, and OWLQN
   * ([[http://www.scalanlp.org/api/breeze/index.html#breeze.optimize.OWLQN breeze.optimize.OWLQN]]) optimizer from
   * Breeze to optimize functions with L1 penalty term. The diffFunction is also translated to a form these breeze
   * optimizer can understand.
   * The L1 penalty is implemented in the optimizer level. See
   * [[http://www.scalanlp.org/api/breeze/index.html#breeze.optimize.OWLQN breeze.optimize.OWLQN]].
   */
  protected[ml] class BreezeOptimization(
      data: Either[RDD[Datum], Iterable[Datum]],
      diffFunction: DiffFunction[Datum],
      initialCoef: Vector[Double]) {

    private val lbfgs = diffFunction match {
      case diffFunc: DiffFunction[Datum] with L1RegularizationTerm =>
        val l1Weight = diffFunc.getL1RegularizationParam
        new BreezeOWLQN[Int, Vector[Double]](maxNumIterations, numCorrections, (_: Int) => l1Weight, tolerance)
      case diffFunc: DiffFunction[Datum] =>
        new BreezeLBFGS[Vector[Double]](maxNumIterations, numCorrections, tolerance)
    }
    private val breezeDiffFunction = new BreezeDiffFunction[Vector[Double]]() {
      //calculating the gradient and value of the objective function
      override def calculate(coefficients: Vector[Double]): (Double, Vector[Double]) = {
        data match {
          //the calculation will be done in a distributed fashion
          case Left(dataAsRDD) =>
            val broadcastedCoefficients = dataAsRDD.context.broadcast(coefficients)
            val (value, gradient) = diffFunction.calculate(dataAsRDD, broadcastedCoefficients)
            broadcastedCoefficients.unpersist()
            (value, gradient)
          //the calculation will be done on a local machine.
          case Right(dataAsIterable) => diffFunction.calculate(dataAsIterable, coefficients)
        }
      }
    }
    private val breezeStates = lbfgs.iterations(breezeDiffFunction, initialCoef)
    breezeStates.next()

    def next(state: OptimizerState): OptimizerState = {
      if (breezeStates.hasNext) {
        val breezeState = breezeStates.next()
        /* project coefficients into constrained space, if any, before updating the state */
        OptimizerState(
          OptimizationUtils.projectCoefficientsToHypercube(breezeState.x, constraintMap), breezeState.adjustedValue,
            breezeState.adjustedGradient, state.iter + 1)
      } else {
        //lbfgs is converged
        state
      }
    }
  }

  @transient private var breezeOptimization: BreezeOptimization = _

  /**
   * Initialize breeze optimization engine.
   * @param state The optimizer state for validation, debugging and logging purposes
   * @param data Input data
   * @param diffFunction The loss function to be optimized
   * @param initialCoef Initial coefficients for the optimization
   */
  def init(
      state: OptimizerState,
      data: Either[RDD[Datum], Iterable[Datum]],
      diffFunction: DiffFunction[Datum],
      initialCoef: Vector[Double]) {
    breezeOptimization = new BreezeOptimization(data, diffFunction, initialCoef)
  }

  override def clearOptimizerInnerState() {
    breezeOptimization = _:BreezeOptimization
  }

  protected def runOneIteration(
      data: Either[RDD[Datum], Iterable[Datum]],
      objectiveFunction: DiffFunction[Datum],
      state: OptimizerState): OptimizerState = {
    breezeOptimization.next(state)
  }
}

object LBFGS {
  val DEFAULT_NUM_CORRECTIONS = 10
  val DEFAULT_MAX_ITER = 80
  val DEFAULT_TOLERANCE = 1.0E-7
}
