package com.linkedin.photon.ml.normalization

import breeze.linalg.{DenseVector, SparseVector, Vector}
import com.linkedin.mlease.spark.stat.BasicStatistics
import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.stat.{BasicStatistics, BasicStatisticalSummary}
import org.apache.spark.rdd.RDD


/**
 * Transform an input vector by rescaling individual components. Used to implement [[NormalizationType]].USE_STANDARD_DEVIATION
 * and [[NormalizationType]].USE_MAX_MAGNITUDE.
 * @param factors the factor vector to do the scaling
 * @author dpeng
 */
@SerialVersionUID(1L)
class VectorScaler(factors: Vector[Double]) extends Transformer[Vector[Double]] {
  /**
   * Transform/Scale a vector. The sparsity of the vector is preserved. The output vector is
   * a vector of element-wise multiplication of factors and input vector.
   * @param input Input vector
   * @return Output vector
   */
  override def transform(input: Vector[Double]): Vector[Double] = {
    require(factors.size == input.size, "Vector size and the scaling factor size are different.")
    input match {
      case dv: DenseVector[Double] =>
        input :* factors
      case sv: SparseVector[Double] =>
        // SparseVector :* DenseVector returns a DenseVector.
        // To preserve the sparseness, we should use indices explicitly to do the transformation.
        val indices = sv.index.clone()
        val values = indices.zip(sv.data).map {
          case (index, value) =>
            value * factors(index)
        }
        new SparseVector[Double](indices, values, input.size)
      case v => throw new IllegalArgumentException("Do not support vector type " + v.getClass)
    }
  }
}



object VectorTransformerFactory {
  /**
   * Create a transformer that implements [[NormalizationType]].USE_STANDARD_DEVIATION. All features are scaled to unit
   * variance. Note: features with zero variance are untouched.
   * @param summary The summary to provide information for the scaling
   * @return A vector scaler to scale features to unit variance
   */
  private def getStandardDeviationScaler(summary: BasicStatisticalSummary): VectorScaler = {
    val factors = summary.variance.map(x => {
      val std = math.sqrt(x)
      if (std == 0) 1.0 else 1.0 / std
    })
    new VectorScaler(factors)
  }

  /**
   * Create a transformer that implements [[NormalizationType]].USE_MAX_MAGNITUDE. All features are scaled to the range of
   * [-1, 1]. Note: Features with all zero values are untouched.
   * @param summary The summary to provide information for the scaling
   * @return A vector scaler to scale features to the range of [-1, 1]
   */
  private def getMaxMagnitudeScaler(summary: BasicStatisticalSummary): VectorScaler = {
    val factors = summary.max.toArray.zip(summary.min.toArray).map {
      case (max, min) =>
        val magnitude = math.max(math.abs(max), math.abs(min))
        if (magnitude == 0) 1.0 else 1.0 / magnitude
    }
    new VectorScaler(DenseVector(factors))
  }

  /**
   * Get a vector [[Transformer]] according to a normalization type and the input RDD.
   * @param normalizationType The normalization type
   * @param inputRdd The input data RDD
   * @return The vector transformer
   */
  def apply(normalizationType: NormalizationType, inputRdd: RDD[LabeledPoint]): Transformer[Vector[Double]] = {
    //  Not all normalization (e.g. NO_SCALING) requires summary so summary is lazy evaluated.
    lazy val summary = BasicStatistics.getBasicStatistics(inputRdd)
    apply(normalizationType, summary)
  }

  /**
   * Get a vector [[Transformer]] according to a normalization type and the feature summary.
   * @param normalizationType The normalization type
   * @param summary The feature summary. Not all normalization (e.g. NO_SCALING) requires summary so summary is a call-by-name argument.
   * @return  The vector transformer
   */
  def apply(normalizationType: NormalizationType, summary: => BasicStatisticalSummary): Transformer[Vector[Double]] = {
    normalizationType match {
      case NormalizationType.USE_STANDARD_DEVIATION =>
        getStandardDeviationScaler(summary)
      case NormalizationType.USE_MAX_MAGNITUDE =>
        getMaxMagnitudeScaler(summary)
      case NormalizationType.NO_SCALING =>
        new IdentityTransformer[Vector[Double]]
      case _ =>
        throw new IllegalArgumentException(s"Normalization type $normalizationType not recognized")
    }
  }
}
