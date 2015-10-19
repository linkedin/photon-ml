package com.linkedin.photon.ml.io

/**
 * A trait that defines the common methods for implementing a factory that provides builders for training inputs.
 * Different formats might implement it different. This is intended to be used in tests.
 *
 * @author yizhou
 */
trait TrainingAvroBuilderFactory {
  def newBuilder(): TrainingAvroBuilder
}
