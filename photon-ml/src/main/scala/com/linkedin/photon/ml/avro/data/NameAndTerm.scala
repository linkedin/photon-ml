package com.linkedin.photon.ml.avro.data

/**
 * @author xazhang
 */
case class NameAndTerm(name: String, term: String) {

  override def hashCode: Int = {
    (name + NameAndTerm.DELIMITER + term).hashCode
  }

  override def toString: String = {
    s"name: $name, term: $term"
  }
}

protected[avro] object NameAndTerm {
  private val DELIMITER = "\u0000"

  val INTERCEPT_NAME_AND_TERM = NameAndTerm("(INTERCEPT)", "")
}
