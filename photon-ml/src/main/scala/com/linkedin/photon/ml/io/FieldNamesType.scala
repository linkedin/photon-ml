package com.linkedin.photon.ml.io

/**
 * Supported types of field names
 * @author xazhang
 */
object FieldNamesType extends Enumeration {
  type FieldNamesType = Value
  val RESPONSE_PREDICTION, TRAINING_EXAMPLE, NONE = Value
}
