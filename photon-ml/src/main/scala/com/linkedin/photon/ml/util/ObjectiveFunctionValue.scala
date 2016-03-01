package com.linkedin.photon.ml.util

/**
 * @author xazhang
 */
case class ObjectiveFunctionValue(lossFunctionValue: Double, regularizationTermValue: Double) {
  val objectiveFunctionValue: Double = lossFunctionValue + regularizationTermValue

  override def toString: String = {
    s"lossFunctionValue: $lossFunctionValue, " +
      s"regularizationTermValue: $regularizationTermValue, " +
      s"objectiveFunctionValue: $objectiveFunctionValue"
  }
}
