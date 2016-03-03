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
package com.linkedin.photon.ml.function

import breeze.linalg.Vector
import com.linkedin.photon.ml.data.DataPoint
import com.linkedin.photon.ml.optimization.{LBFGS, RegularizationContext}
import com.linkedin.photon.ml.util.Utils
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD

/**
 * Trait for twice differentiable function. Similar to a differentiable function, the value/gradient/hessian of
 * a twice differentiable function depends two type of variables, the input datum/data and model coefficients
 * (e.g., the weighs of the features in a generalized linear model).
 * @tparam Datum Generic type of data point
 * @author xazhang
 */

trait TwiceDiffFunction[Datum <: DataPoint] extends DiffFunction[Datum] {

  /**
   * First calculate the Hessian of the function given one data point and model coefficients, then multiply it with a
   * given vector
   * @param datum The given datum at which point to compute the hessian multiplied by a given vector
   * @param coefficients The given model coefficients used to compute the hessian multiplied by a given vector
   * @param multiplyVector The given multiplyVector to be multiplied with the Hessian. For example, in conjugate
   *                       gradient method this multiplyVector would correspond to the gradient multiplyVector.
   * @return The computed Hessian multiplied by the given multiplyVector
   */
  protected[ml] def hessianVectorAt(
      datum: Datum,
      coefficients: Vector[Double],
      multiplyVector: Vector[Double]): Vector[Double] = {

    val cumHessianVector = Utils.initializeZerosVectorOfSameType(coefficients)
    hessianVectorAt(datum, coefficients, multiplyVector, cumHessianVector)
    cumHessianVector
  }

  /**
   * First calculate the Hessian of the function under given one data point and model coefficients, then multiply it
   * with a given multiplyVector and add to cumGradient in place.
   * @param datum The given datum at which point to compute the hessian multiplied by a given vector
   * @param coefficients The given model coefficients used to compute the hessian multiplied by a given vector
   * @param multiplyVector The given multiplyVector to be multiplied with the Hessian. For example, in conjugate
   *                       gradient method this multiplyVector would correspond to the gradient multiplyVector.
   * @param cumHessianVector The cumulative sum of the previously computed Hessian multiplyVector
   */
  protected[ml] def hessianVectorAt(
    datum: Datum,
    coefficients: Vector[Double],
    multiplyVector: Vector[Double],
    cumHessianVector: Vector[Double]): Unit

  /**
   * Compute the Hessian of the function under the given data and coefficients, then multiply it with a given
   * multiplyVector.
   * @param data The given data at which point to compute the hessian multiplied by a given vector
   * @param broadcastedCoefficients The broadcasted model coefficients used to compute the hessian multiplied by a given
   *                                vector
   * @param multiplyVector The given multiplyVector to be multiplied with the Hessian. For example, in conjugate
   *                       gradient method this multiplyVector would correspond to the gradient multiplyVector.
   * @return The computed Hessian multiplied by the given multiplyVector
   */
  protected[ml] def hessianVector(
      data: RDD[Datum],
      broadcastedCoefficients: Broadcast[Vector[Double]],
      multiplyVector: Broadcast[Vector[Double]]): Vector[Double] = {

    val initialCumHessianVector = Utils.initializeZerosVectorOfSameType(broadcastedCoefficients.value)
    data.treeAggregate(initialCumHessianVector)(
      seqOp = (cumHessianVector, datum) => {
        hessianVectorAt(datum, broadcastedCoefficients.value, multiplyVector.value, cumHessianVector)
        cumHessianVector
      },
      combOp = _ += _,
      depth = treeAggregateDepth
    )
  }

  /**
   * Compute the Hessian of the function under the given data and coefficients, then multiply it with a given
   * multiplyVector.
   * @param data The given data at which point to compute the hessian multiplied by a given vector
   * @param coefficients The given model coefficients used to compute the hessian multiplied by a given vector
   * @param multiplyVector The given multiplyVector to be multiplied with the Hessian. For example, in conjugate
   *                       gradient method this multiplyVector would correspond to the gradient multiplyVector.
   * @return The computed Hessian multiplied by the given multiplyVector
   */
  protected[ml] def hessianVector(
      data: Iterable[Datum],
      coefficients: Vector[Double],
      multiplyVector: Vector[Double]): Vector[Double] = {

    val initialCumHessianVector = Utils.initializeZerosVectorOfSameType(coefficients)
    data.aggregate(initialCumHessianVector)(
      seqop = (cumHessianVector, datum) => {
        hessianVectorAt(datum, coefficients, multiplyVector, cumHessianVector)
        cumHessianVector
      },
      combop = _ += _
    )
  }
}

object TwiceDiffFunction {
  /**
   * An anonymous class for the twice differentiable function with L2 regularization
   * @param func The twice differential function.
   * @param regWeight The weight for the regularization term.
   * @tparam Datum The generic type of the datum
   * @return An anonymous class for the twice differentiable function with L2 regularization
   */
  private def withL2Regularization[Datum <: DataPoint](func: TwiceDiffFunction[Datum], regWeight: Double) =
      new TwiceDiffFunction[Datum] {

    override protected[ml] def calculateAt(
        datum: Datum,
        coefficients: Vector[Double],
        cumGradient: Vector[Double]): Double = {

      val v = func.calculateAt(datum, coefficients, cumGradient)
      cumGradient += gradientOfL2Reg(coefficients)
      v + valueOfL2Reg(coefficients)
    }

    override protected[ml] def hessianVectorAt(
        datum: Datum,
        coefficients: Vector[Double],
        multiplyVector: Vector[Double],
        cumHessianVector: Vector[Double]): Unit = {
      func.hessianVectorAt(datum, coefficients, multiplyVector, cumHessianVector)
      cumHessianVector += hessianVectorOfL2Reg(multiplyVector)
    }

    override protected[ml] def calculate(
        data: RDD[Datum],
        broadcastedCoefficients: Broadcast[Vector[Double]]): (Double, Vector[Double]) = {

      val (v, grad) = func.calculate(data, broadcastedCoefficients)
      (v + valueOfL2Reg(broadcastedCoefficients.value), grad + gradientOfL2Reg(broadcastedCoefficients.value))
    }

    override protected[ml] def calculate(
        data: Iterable[Datum],
        coefficients: Vector[Double]): (Double, Vector[Double]) = {

      val (v, grad) = func.calculate(data, coefficients)
      (v + valueOfL2Reg(coefficients), grad + gradientOfL2Reg(coefficients))
    }

    override protected[ml] def hessianVector(
        data: RDD[Datum],
        broadcastedCoefficients: Broadcast[Vector[Double]],
        multiplyVector: Broadcast[Vector[Double]]): Vector[Double] = {
      func.hessianVector(data, broadcastedCoefficients, multiplyVector) + hessianVectorOfL2Reg(multiplyVector.value)
    }

    override protected[ml] def hessianVector(
        data: Iterable[Datum],
        coefficients: Vector[Double],
        multiplyVector: Vector[Double]): Vector[Double] = {
      func.hessianVector(data, coefficients, multiplyVector) + hessianVectorOfL2Reg(multiplyVector)
    }

    private def valueOfL2Reg(coefficients: Vector[Double]) = {
      regWeight * (coefficients dot coefficients) / 2
    }

    private def gradientOfL2Reg(coefficients: Vector[Double]): Vector[Double] = {
      coefficients * regWeight
    }

    private def hessianVectorOfL2Reg(multiplyVector: Vector[Double]) = {
      multiplyVector * regWeight
    }
  }

  /**
   * An anonymous class for the twice differentiable function with L1 regularization. The only effect of this binding is
   * to label the function with the L1 regularization weight, with all function values, gradients, Hessian untouched.
   * @param func The twice differential function.
   * @param regWeight The weight for the regularization term.
   * @tparam Datum The generic type of the datum
   * @return An anonymous class for the twice differentiable function with L1 regularization
   */
  private def withL1Regularization[Datum <: DataPoint](
      func: TwiceDiffFunction[Datum],
      regWeight: Double): TwiceDiffFunction[Datum]
    with L1RegularizationTerm = new TwiceDiffFunction[Datum] with L1RegularizationTerm {

    override protected[ml] def calculateAt(
        datum: Datum,
        coefficients: Vector[Double],
        cumGradient: Vector[Double]): Double = {
      func.calculateAt(datum, coefficients, cumGradient)
    }

    override protected[ml] def hessianVectorAt(
        datum: Datum,
        coefficients: Vector[Double],
        multiplyVector: Vector[Double],
        cumHessianVector: Vector[Double]): Unit = {
      func.hessianVectorAt(datum, coefficients, multiplyVector, cumHessianVector)
    }

    override protected[ml] def calculate(
        data: RDD[Datum],
        broadcastedCoefficients: Broadcast[Vector[Double]]): (Double, Vector[Double]) = {
      func.calculate(data, broadcastedCoefficients)
    }

    override protected[ml] def calculate(
        data: Iterable[Datum],
        coefficients: Vector[Double]): (Double, Vector[Double]) = {
      func.calculate(data, coefficients)
    }

    override protected[ml] def hessianVector(
        data: RDD[Datum],
        broadcastedCoefficients: Broadcast[Vector[Double]],
        multiplyVector: Broadcast[Vector[Double]]): Vector[Double] = {
      func.hessianVector(data, broadcastedCoefficients, multiplyVector)
    }

    override protected[ml] def hessianVector(
        data: Iterable[Datum],
        coefficients: Vector[Double],
        multiplyVector: Vector[Double]): Vector[Double] = {
      func.hessianVector(data, coefficients, multiplyVector)
    }

    override def getL1RegularizationParam: Double = regWeight
  }

  /**
   * Add regularization to the twice differentiable function. Under the hood, the L2 regularization part is added to the
   * loss function values/gradient/Hessian, but the L1 regularization has only a regularization weight to be further
   * used by the optimizer (especially
   * [[http://www.scalanlp.org/api/breeze/index.html#breeze.optimize.OWLQN breeze.optimize.OWLQN]] used in
   * [[LBFGS LBFGS]]).
   * @param func The differentiable function
   * @param regularizationContext The regularization context
   * @param regWeight The regularization weight
   * @tparam Datum Datum type
   * @return The twice differentiable function with necessary decorations
   */
  def withRegularization[Datum <: DataPoint](
      func: TwiceDiffFunction[Datum],
      regularizationContext: RegularizationContext,
      regWeight: Double): TwiceDiffFunction[Datum] = {

    val (l1Weight, l2Weight) = (
      regularizationContext.getL1RegularizationWeight(regWeight),
      regularizationContext.getL2RegularizationWeight(regWeight))

    (l1Weight,  l2Weight) match {
      case (0.0, 0.0) =>
        // No regularization
        func
      case (0.0, l2) =>
        // Only l2 regularization
        withL2Regularization(func, l2)
      case (l1, 0.0) =>
        // Only l1 regularization
        withL1Regularization(func, l1)
      case (l1, l2) =>
        // L1 and L2 regularization
        val funcWithL2 = withL2Regularization(func, l2)
        withL1Regularization(funcWithL2, l1)
    }
  }
}
