package com.linkedin.photon.ml

/**
 * This is an Enum class that enumerates all stages of Photon-ML Driver
 *
 * @author yizhou
 */
class DriverStage private(val name: String, val order: Int) {
  def ==(that: DriverStage): Boolean = {
    order == that.order
  }

  def <(that: DriverStage): Boolean = {
    order < that.order
  }

  def >(that: DriverStage): Boolean = {
    order > that.order
  }

  def <=(that: DriverStage): Boolean = {
    order <= that.order
  }

  def >=(that: DriverStage): Boolean = {
    order >= that.order
  }

  override def toString: String = {
    "{name: " + name + ", order: " + order + "}"
  }
}

object DriverStage extends Enumeration {
  val INIT = new DriverStage("INIT", 0)
  val PREPROCESSED = new DriverStage("PREPROCESSED", 1)
  val TRAINED = new DriverStage("TRAINED", 2)
  val VALIDATED = new DriverStage("VALIDATED", 3)
}
