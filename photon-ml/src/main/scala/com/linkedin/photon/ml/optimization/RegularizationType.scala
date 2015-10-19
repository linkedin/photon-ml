package com.linkedin.photon.ml.optimization

/**
 * Supported regularization types
 * @author xazhang
 * @author dpeng
 */
object RegularizationType extends Enumeration {
  type RegularizationType = Value
  val L2, L1, ELASTIC_NET, NONE = Value
}
