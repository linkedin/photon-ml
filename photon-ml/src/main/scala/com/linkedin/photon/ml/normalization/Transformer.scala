package com.linkedin.photon.ml.normalization

/**
 * Apply some transformation to the input instance.
 */
trait Transformer[T] extends Serializable {
  def transform(input: T): T
}
