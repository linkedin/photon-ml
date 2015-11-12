package com.linkedin.photon.ml

/**
 * Control the level of validation that is performed
 * @author bdrew
 */
object DataValidationType extends Enumeration {
  type DataValidationType = Value
  val VALIDATE_FULL, VALIDATE_SAMPLE, VALIDATE_DISABLED = Value
}
