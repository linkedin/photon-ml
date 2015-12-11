package com.linkedin.photon.ml.function

import breeze.linalg.Vector
import com.linkedin.photon.ml.data.DataPoint
import com.linkedin.photon.ml.optimization.{LBFGS, RegularizationContext}
import com.linkedin.photon.ml.util.Utils
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD


/**
 * Trait for differentiable function. The value/gradient of a differentiable function depends on
 * two types of variables, data (a data point or a RDD of data points) and model coefficients
 * (e.g., the coefficients learned in a generalized linear model).
 *
 * FAQ 1: Why the function takes BroadCast[ Vector[Double] ] typed input coefficients instead of Vector[Double] when
 * computing the function value and gradient for a RDD of input data points?
 * A: In the optimization procedure, the gradient/value/calculate function might be called repeatedly several times
 * given the same input coefficients (e.g., during conjugate gradient method). As a result, if the input
 * coefficients type is Vector, we need to re-broadcast the input vector every time to make it accessible in the function,
 * even when the value of the input vector itself is unchanged. This would potentially cause some network I/O problem
 * if this function is called frequently, and would result in a waste of computation as well. However, if the input
 * coefficients is of type BroadCast[ Vector[Double] ], we can re-use it for different function calls as long as the
 * underlying vector hasn't changed.
 *
 * FAQ 2: Why use cumGradient (cumulative gradient)?
 * A: Using cumGradient would allow the functions to modify and then return cumGradient instead of creating a new gradient
 * vector to avoid memory allocation.
 *
 * @tparam Datum Generic type of data point
 * @author xazhang
 */

trait DiffFunction[Datum <: DataPoint] extends Serializable {

  /**
   * Calculate the gradient of the function given one data point and the model coefficients
   * @param datum The given datum at which point to compute the gradient
   * @param coefficients The given model coefficients used to compute the gradient
   * @return The computed gradient of the function
   */
  protected[ml] def gradientAt(datum: Datum, coefficients: Vector[Double]): Vector[Double] = {
    calculateAt(datum, coefficients)._2
  }

  /**
   * Calculate the value of the function given one data point and model coefficients
   * @param datum The given datum at which point to compute the function's value
   * @param coefficients The given model coefficients used to compute the function's value
   * @return The computed value of the function
   */
  protected[ml] def valueAt(datum: Datum, coefficients: Vector[Double]): Double = {
    calculateAt(datum, coefficients)._1
  }

  /**
   * Calculate both the value and the gradient of the function given one data point and model coefficients
   * @param datum The given datum at which point to compute the function's value and gradient
   * @param coefficients The given model coefficients used to compute the function's value and gradient
   * @return The computed value and gradient of the function
   */
  protected[ml] def calculateAt(datum: Datum, coefficients: Vector[Double]): (Double, Vector[Double]) = {
    val cumGradient = Utils.initializeZerosVectorOfSameType(coefficients)
    val value = calculateAt(datum, coefficients, cumGradient)
    (value, cumGradient)
  }

  /**
   * Calculate both the value and the gradient of the function given one data point and model coefficients, with
   * the computed gradient added to cumGradient in place.
   * @param datum The given datum at which point to compute the function's value and gradient
   * @param coefficients The given model coefficients used to compute function's value and gradient
   * @param cumGradient The cumulative gradient
   * @return The computed value of the function
   */
  protected[ml] def calculateAt(datum: Datum, coefficients: Vector[Double], cumGradient: Vector[Double]): Double

  /**
   * Calculate the gradient of the function given a RDD of data points and model coefficients
   * @param data The given data at which point to compute the function's gradient
   * @param broadcastedCoefficients The broadcasted model coefficients used to compute the function's gradient
   * @return The computed gradient of the function
   */
  protected[ml] def gradient(data: RDD[Datum], broadcastedCoefficients: Broadcast[Vector[Double]]): Vector[Double] = {
    calculate(data, broadcastedCoefficients)._2
  }

  /**
   * Calculate the value of the function given a RDD of data point and model coefficients
   * @param data The given data at which point to compute the function's value
   * @param broadcastedCoefficients The broadcasted model coefficients used to compute the function's value
   * @return The computed value of the function
   */
  protected[ml] def value(data: RDD[Datum], broadcastedCoefficients: Broadcast[Vector[Double]]): Double = {
    calculate(data, broadcastedCoefficients)._1
  }

  /**
   * Calculate both the value and the gradient of the function given a RDD of data points and model coefficients
   * (compute value and gradient at once is sometimes more efficient than computing them sequentially)
   * @param data The given data at which point to compute the function's value and gradient
   * @param broadcastedCoefficients The broadcasted model coefficients used to compute the function's value and gradient
   * @return The computed value and gradient of the function
   */
  protected[ml] def calculate(data: RDD[Datum], broadcastedCoefficients: Broadcast[Vector[Double]]): (Double, Vector[Double]) = {
    val initialCumGradient = Utils.initializeZerosVectorOfSameType(broadcastedCoefficients.value)
    data.aggregate((0.0, initialCumGradient))(
      seqOp = {
        case ((value, cumGradient), datum) =>
          val v = calculateAt(datum, broadcastedCoefficients.value, cumGradient)
          (value + v, cumGradient)
      },
      combOp = {
        case ((loss1, grad1), (loss2, grad2)) =>
          (loss1 + loss2, grad1 += grad2)
      })
  }

  /**
   * Calculate the gradient of the function given a sequence of data points and model coefficients
   * @param data The given data at which point to compute the function's gradient
   * @param coefficients The given model coefficients used to compute the gradient
   * @return The computed gradient of the function
   */
  protected[ml] def gradient(data: Iterable[Datum], coefficients: Vector[Double]): Vector[Double] = {
    calculate(data, coefficients)._2
  }

  /**
   * Calculate the value of the function given a sequence of data point and model coefficients
   * @param data The given data at which point to compute the function's value
   * @param coefficients The given model coefficients used to compute the function's value
   * @return The computed value of the function
   */
  def value(data: Iterable[Datum], coefficients: Vector[Double]): Double = {
    calculate(data, coefficients)._1
  }

  /**
   * Calculate both the value and the gradient of the function given a sequence of data points and model coefficients
   * (compute value and gradient at once is sometimes more efficient than computing them sequentially)
   * @param data The given data at which point to compute the function's value and gradient
   * @param coefficients The given model coefficients used to compute the function's value and gradient
   * @return The computed value and gradient of the function
   */
  protected[ml] def calculate(data: Iterable[Datum], coefficients: Vector[Double]): (Double, Vector[Double]) = {
    val initialCumGradient = Utils.initializeZerosVectorOfSameType(coefficients)
    data.aggregate((0.0, initialCumGradient))(
      seqop = {
        case ((loss, cumGradient), datum) =>
          val v = calculateAt(datum, coefficients, cumGradient)
          (loss + v, cumGradient)
      },
      combop = {
        case ((loss1, grad1), (loss2, grad2)) =>
          (loss1 + loss2, grad1 += grad2)
      })
  }

  /**
   * Compute the margin of the data to the decision boundary, which is defined as the dot product between the
   * feature vector of the data and the coefficients vector
   * @param datum The given datum at which point to compute the margin
   * @param coefficients The given model coefficients used to compute the margin
   * @return The computed margin
   */
  protected def computeMargin(datum: Datum, coefficients: Vector[Double]): Double = {
    datum.computeMargin(coefficients)
  }
}

object DiffFunction {
  /**
   * An anonymous class for differentiable function with L2 regularization
   * @param func The differentiable function.
   * @param regWeight The weight for the regularization term.
   * @tparam Datum Generic type of data point
   * @return An anonymous class for differentiable function with L2 regularization
   */
  private def withL2Regularization[Datum <: DataPoint](func: DiffFunction[Datum], regWeight: Double) = new DiffFunction[Datum] {

    override protected[ml] def calculateAt(data: Datum, coefficients: Vector[Double], cumGradient: Vector[Double]): Double = {
      val v = func.calculateAt(data, coefficients, cumGradient)
      cumGradient += gradientOfL2Reg(coefficients)
      v + valueOfL2Reg(coefficients)
    }

    override protected[ml] def calculate(data: RDD[Datum], broadcastedCoefficients: Broadcast[Vector[Double]]): (Double, Vector[Double]) = {
      val (v, grad) = func.calculate(data, broadcastedCoefficients)
      (v + valueOfL2Reg(broadcastedCoefficients.value), grad + gradientOfL2Reg(broadcastedCoefficients.value))
    }

    override protected[ml] def calculate(data: Iterable[Datum], coefficients: Vector[Double]): (Double, Vector[Double]) = {
      val (v, grad) = func.calculate(data, coefficients)
      (v + valueOfL2Reg(coefficients), grad + gradientOfL2Reg(coefficients))
    }

    private def valueOfL2Reg(coefficients: Vector[Double]) = {
      regWeight * (coefficients dot coefficients) / 2
    }

    private def gradientOfL2Reg(coefficients: Vector[Double]): Vector[Double] = {
      coefficients * regWeight
    }
  }

  /**
   * An anonymous class for the differentiable function with L1 regularization. The only effect of this binding is
   * to label the function with the L1 regularization weight, with all function values, gradients, Hessian untouched.
   * @param func The differential function.
   * @param regWeight The weight for the regularization term.
   * @tparam Datum The generic type of the datum
   * @return An anonymous class for the twice differentiable function with L1 regularization
   */
  private def withL1Regularization[Datum <: DataPoint](func: DiffFunction[Datum], regWeight: Double): DiffFunction[Datum] with L1RegularizationTerm = new DiffFunction[Datum] with L1RegularizationTerm {

    override protected[ml] def calculateAt(datum: Datum,
                                              coefficients: Vector[Double],
                                              cumGradient: Vector[Double]): Double = {
      func.calculateAt(datum, coefficients, cumGradient)
    }

    override protected[ml] def calculate(data: RDD[Datum], broadcastedCoefficients: Broadcast[Vector[Double]]): (Double, Vector[Double]) = {
      func.calculate(data, broadcastedCoefficients)
    }

    override protected[ml] def calculate(data: Iterable[Datum], coefficients: Vector[Double]): (Double, Vector[Double]) = {
      func.calculate(data, coefficients)
    }
    override def getL1RegularizationParam: Double = regWeight
  }

  /**
   * Add regularization to the differentiable function. Under the hood, the L2 regularization part is added to the loss
   * function values/gradient/Hessian, but the L1 regularization has only a regularization weight to be further used by
   * the optimizer (especially
   * [[[http://www.scalanlp.org/api/breeze/index.html#breeze.optimize.OWLQN breeze.optimize.OWLQN]] used in
   * [[LBFGS LBFGS]]).
   * @param func The differentiable function
   * @param regularizationContext The regularization context
   * @param regWeight The regularization weight
   * @tparam Datum Datum type
   * @return The differentiable function with necessary decorations
   */
  def withRegularization[Datum <: DataPoint](func: DiffFunction[Datum], regularizationContext: RegularizationContext, regWeight: Double): DiffFunction[Datum] = {
    val (l1Weight, l2Weight) = (regularizationContext.getL1RegularizationWeight(regWeight), regularizationContext.getL2RegularizationWeight(regWeight))
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

