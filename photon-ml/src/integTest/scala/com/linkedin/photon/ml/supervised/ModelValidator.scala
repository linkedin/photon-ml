package com.linkedin.photon.ml.supervised

import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

/**
 * Encapsulates the logic for validating a model fit via an instance of
 * {@link com.linkedin.photon.ml.supervised.GeneralizedLinearAlgorithm}.
 *
 * It is expected that validateModel will be called before validateModelPredictions to try to
 * allow for fast(er) failure.
 */
abstract class ModelValidator[-GLM <: GeneralizedLinearModel : ClassTag] extends Serializable {

  /**
   * Inspect a model's predictions and determine whether they are "sensible" / "valid"
   *
   * Should throw some reasonable kind of exception if this is _not_ the case.
   * @param model
   * @param data
   */
  def validateModelPredictions(model:GLM, data:RDD[LabeledPoint]): Unit
}
