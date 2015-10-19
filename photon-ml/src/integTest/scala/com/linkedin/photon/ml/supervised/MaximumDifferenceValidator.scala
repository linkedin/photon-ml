package com.linkedin.photon.ml.supervised

import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import com.linkedin.photon.ml.supervised.regression.Regression
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

/**
 * Created by bdrew on 9/24/15.
 */
class MaximumDifferenceValidator[-R <: GeneralizedLinearModel with Regression with Serializable: ClassTag](maximumDifference:Double) extends ModelValidator[R] {
  assert(maximumDifference > 0.0)
  def validateModelPredictions(model:R, data:RDD[LabeledPoint]) = {
    val countTooBig = data.filter( x => { Math.abs(model.predict(x.features) - x.label) > maximumDifference}).count
    if (countTooBig > 0) {
      throw new IllegalStateException(s"Found [$countTooBig] instances where the magnitude of the prediction error is greater than [$maximumDifference]")
    }
  }
}
