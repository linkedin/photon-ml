package com.linkedin.photon.ml.io

import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel

/**
 * Field names of the Avro formatted file used as input of [[GeneralizedLinearModel]]
 * @author xazhang
 */
trait FieldNames extends Serializable {
  val features: String
  val name: String
  val term: String
  val value: String
  val response: String
  val offset: String
  val weight: String
}
