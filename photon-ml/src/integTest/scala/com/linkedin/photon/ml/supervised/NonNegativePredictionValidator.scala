package com.linkedin.photon.ml.supervised

import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import com.linkedin.photon.ml.supervised.regression.Regression
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

/**
 * Verify that on a particular data set, the model only produces nonnegative predictions
 *
 * @author asaha
 */
class NonNegativePredictionValidator[-GLM <: GeneralizedLinearModel with Regression: ClassTag] extends ModelValidator[GLM] {

  override def validateModelPredictions(model:GLM, data:RDD[LabeledPoint]) : Unit = {
    val predictions = model.predictAll(data.map(x => x.features))
    val invalidCount = predictions.filter(x => x < 0).count
    if (invalidCount > 0) {
      throw new IllegalStateException(s"Found [$invalidCount] samples with invalid predictions (expect non-negative labels only).")
    }
  }
}
