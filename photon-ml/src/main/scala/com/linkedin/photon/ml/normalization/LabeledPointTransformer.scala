package com.linkedin.photon.ml.normalization

import breeze.linalg.Vector
import com.linkedin.photon.ml.data
import com.linkedin.photon.ml.data.LabeledPoint

/**
 * A scaler to scale [[http://www.scalanlp.org/api/breeze/#breeze.linalg.Vector breeze.linalg.Vector]] in the
 * [[data.LabeledPoint LabeledPoint]].
 * @author dpeng
 */
@SerialVersionUID(1L)
class LabeledPointTransformer(vectorTransformer: Transformer[Vector[Double]]) extends Transformer[LabeledPoint] {
  /**
   * Transform/Scale a [[data.LabeledPoint LabeledPoint]]. It only transforms the underlying
   * vector in the [[data.LabeledPoint LabeledPoint]]. The sparsity of the vector is
   * preserved.
   * @param input Input [[data.LabeledPoint LabeledPoint]]
   * @return Output [[data.LabeledPoint LabeledPoint]]
   */
  override def transform(input: LabeledPoint): LabeledPoint = {
    val LabeledPoint(label, features, weight, offset) = input
    val transformedFeatures = vectorTransformer.transform(features)
    LabeledPoint(label, transformedFeatures, weight, offset)
  }
}
