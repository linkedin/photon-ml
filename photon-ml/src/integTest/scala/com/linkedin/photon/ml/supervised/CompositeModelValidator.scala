package com.linkedin.photon.ml.supervised

import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

/**
 * Chain several validators together
 */
class CompositeModelValidator[-GLM <: GeneralizedLinearModel : ClassTag](validators:ModelValidator[GLM]*) extends ModelValidator[GLM] {

  def validateModelPredictions(model:GLM, data:RDD[LabeledPoint]) = {
    validators.foreach(v => { v.validateModelPredictions(model, data) })
  }
}
