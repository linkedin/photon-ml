package com.linkedin.photon.ml.io

/**
 * Enum of the keys that are expected to be present in the maps in the user-specified constraint string
 * The "name" and "term" keys are mandatory while one of "lowerBound" and "upperBound" is allowed to be skipped in
 * which case it is assumed to be -Inf / +Inf appropriately
 *
 * @author nkatariy
 */
object ConstraintMapKeys extends Enumeration {
  type ConstraintMapKeys = Value
  val name, term, lowerBound, upperBound = Value
}