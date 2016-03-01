package com.linkedin.photon.ml.function

import breeze.linalg.Vector
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.data.DataPoint
import com.linkedin.photon.ml.util.Utils

/**
 * Enhanced twice differentiable function with the extra functionality of computing the diagonal elements of the
 * Hessian matrix
 * @author xazhang
 */
trait EnhancedTwiceDiffFunction[Datum <: DataPoint] extends TwiceDiffFunction[Datum] {
  /**
   * Calculate and return the diagonal of the Hessian matrix with a given datum and model coefficients
   * @param datum The given datum at which point to compute the diagonal of the Hessian matrix
   * @param coefficients The given model coefficients used to compute the diagonal of the Hessian matrix
   * @return The computed diagonal of the Hessian matrix
   */
  protected[ml] def hessianDiagonalAt(datum: Datum, coefficients: Vector[Double]): Vector[Double] = {
    val cumHessianDiagonal = Utils.initializeZerosVectorOfSameType(coefficients)
    hessianDiagonalAt(datum, coefficients, cumHessianDiagonal)
    cumHessianDiagonal
  }

  /**
   * First calculate the diagonal of the Hessian matrix with a given datum and model coefficients,
   * then add to cumHessianDiagonal in place.
   * @param datum The given datum at which point to compute the diagonal of the Hessian matrix
   * @param coefficients The given model coefficients used to compute the diagonal of the Hessian matrix
   * @param cumHessianDiagonal The cumulative sum of the previously computed diagonal of the Hessian matrix
   */
  protected[ml] def hessianDiagonalAt(datum: Datum, coefficients: Vector[Double], cumHessianDiagonal: Vector[Double])
  : Unit

  /**
   * Compute the diagonal of Hessian of the function with the given data set and coefficients
   * @param data The given data set with which to compute the diagonal of the Hessian matrix
   * @param coefficientsBroadcast The broadcasted model coefficients used to compute the diagonal of the Hessian matrix
   * @return The computed diagonal of the Hessian matrix
   */
  protected[ml] def hessianDiagonal(data: RDD[Datum], coefficientsBroadcast: Broadcast[Vector[Double]])
  : Vector[Double] = {

    val initialCumHessianDiagonal = Utils.initializeZerosVectorOfSameType(coefficientsBroadcast.value)
    data.treeAggregate(initialCumHessianDiagonal)(
      seqOp = (cumHessianDiagonal, datum) => {
        hessianDiagonalAt(datum, coefficientsBroadcast.value, cumHessianDiagonal)
        cumHessianDiagonal
      },
      combOp = _ += _
    )
  }

  /**
   * Compute the diagonal of Hessian of the function with the given data set and coefficients
   * @param data The given data set with which to compute the diagonal of the Hessian matrix
   * @param coefficients The model coefficients used to compute the diagonal of the Hessian matrix
   * @return The computed diagonal of the Hessian matrix
   */
  protected[ml] def hessianDiagonal(data: Iterable[Datum], coefficients: Vector[Double]): Vector[Double] = {
    val initialCumHessianDiagonal = Utils.initializeZerosVectorOfSameType(coefficients)
    data.aggregate(initialCumHessianDiagonal)(
      seqop = (cumHessianDiagonal, datum) => {
        hessianDiagonalAt(datum, coefficients, cumHessianDiagonal)
        cumHessianDiagonal
      },
      combop = _ += _
    )
  }
}

object EnhancedTwiceDiffFunction {
  /**
   * An anonymous class for the enhanced twice differentiable function with L2 regularization
   * @param func The the twice differential function.
   * @param regWeight The weight for the regularization term.
   * @tparam Datum The generic type of the datum
   * @return An anonymous class for the enhanced twice differentiable function with L2 regularization
   */
  def withL2Regularization[Datum <: DataPoint](func: EnhancedTwiceDiffFunction[Datum], regWeight: Double) =
    new EnhancedTwiceDiffFunction[Datum] {

    override protected[ml] def calculateAt(datum: Datum, coefficients: Vector[Double], cumGradient: Vector[Double])
    : Double = {

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

    override protected[ml] def hessianDiagonalAt(
        datum: Datum,
        coefficients: Vector[Double],
        cumHessianDiagonal: Vector[Double]): Unit = {

      func.hessianDiagonalAt(datum, coefficients, cumHessianDiagonal)
      cumHessianDiagonal += hessianDiagonalOfL2Reg
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

    override protected[ml] def hessianDiagonal(
        data: RDD[Datum],
        broadcastedCoefficients: Broadcast[Vector[Double]]): Vector[Double] = {

      func.hessianDiagonal(data, broadcastedCoefficients) + hessianDiagonalOfL2Reg
    }

    override protected[ml] def hessianDiagonal(
        data: Iterable[Datum],
        coefficients: Vector[Double]): Vector[Double] = {

      func.hessianDiagonal(data, coefficients) + hessianDiagonalOfL2Reg
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

    private def hessianDiagonalOfL2Reg: Double = {
      regWeight
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
  def withL1Regularization[Datum <: DataPoint](func: EnhancedTwiceDiffFunction[Datum], regWeight: Double) =
    new EnhancedTwiceDiffFunction[Datum] with L1RegularizationTerm {

    override protected[ml] def calculateAt(datum: Datum,
                                              coefficients: Vector[Double],
                                              cumGradient: Vector[Double]): Double = {
      func.calculateAt(datum, coefficients, cumGradient)
    }

    override protected[ml] def hessianVectorAt(datum: Datum,
                                                  coefficients: Vector[Double],
                                                  multiplyVector: Vector[Double],
                                                  cumHessianVector: Vector[Double]): Unit = {
      func.hessianVectorAt(datum, coefficients, multiplyVector, cumHessianVector)
    }

    override protected[ml] def hessianDiagonalAt(datum: Datum,
                                                    coefficients: Vector[Double],
                                                    cumHessianDiagonal: Vector[Double]): Unit = {
      func.hessianDiagonalAt(datum, coefficients, cumHessianDiagonal)
    }

    override protected[ml] def calculate(
        data: RDD[Datum],
        broadcastedCoefficients: Broadcast[Vector[Double]]): (Double, Vector[Double]) = {

      func.calculate(data, broadcastedCoefficients)
    }

    override protected[ml] def calculate(data: Iterable[Datum], coefficients: Vector[Double])
    : (Double, Vector[Double]) = {

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

    override protected[ml] def hessianDiagonal(
        data: RDD[Datum],
        broadcastedCoefficients: Broadcast[Vector[Double]]): Vector[Double] = {

      func.hessianDiagonal(data, broadcastedCoefficients)
    }

    override protected[ml] def hessianDiagonal(
        data: Iterable[Datum],
        coefficients: Vector[Double]): Vector[Double] = {

      func.hessianDiagonal(data, coefficients)
    }

    override def getL1RegularizationParam: Double = regWeight
  }
}
