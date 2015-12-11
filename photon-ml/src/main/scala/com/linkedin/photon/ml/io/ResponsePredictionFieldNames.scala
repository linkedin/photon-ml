package com.linkedin.photon.ml.io

/**
 * ADMM's Response prediction format fields name
 * @author xazhang
 */
object ResponsePredictionFieldNames extends FieldNames {
  val features: String = "features"
  val name: String = "name"
  val term: String = "term"
  val value: String = "value"
  val response: String = "response"
  val offset: String = "offset"
  val weight: String = "weight"
}