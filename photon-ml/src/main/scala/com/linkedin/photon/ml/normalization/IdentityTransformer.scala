package com.linkedin.photon.ml.normalization

/**
 * A transformer to transform an object into itself (used to implement NormalizationType.NO_SCALING)
 * @author dpeng
 */
@SerialVersionUID(1L)
class IdentityTransformer[T] extends Transformer[T] {
  override def transform(input: T): T = input
}
