package com.linkedin.photon.ml.io

/**
 * Metronome's TrainingExample format fields name
 * @author xazhang
 */
object TrainingExampleFieldNames extends FieldNames {
  val features: String = "features"
  val name: String = "name"
  val term: String = "term"
  val value: String = "value"
  val response: String = "label"
  val offset: String = "offset"
  val weight: String = "weight"
}
