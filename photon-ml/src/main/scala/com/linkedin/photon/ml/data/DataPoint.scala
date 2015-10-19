package com.linkedin.photon.ml.data

import breeze.linalg.Vector
import com.linkedin.photon.ml.data

/**
 * A general data point contains features and other auxiliary information
 * @param features A vector (could be either dense or sparse) representing the features for this data point
 * @param weight The weight of this data point
 * @author xazhang
 */
class DataPoint(val features: Vector[Double], val weight: Double = 1.0) extends Serializable {
  override def toString: String = {
    s"(features $features\nweight $weight)"
  }
}


/**
 * Companion object of [[data.DataPoint]] for factories and pattern matching purpose
 */
object DataPoint {
  /**
   * Apply methods give you a nice syntactic sugar for when a class or object has one main use.
   */
  def apply(features: Vector[Double], weight: Double): DataPoint = {
    new DataPoint(features, weight)
  }

  /**
   * The extractor
   */
  def unapply(data: DataPoint): Option[(Vector[Double], Double)] =
    Some((data.features, data.weight))
}