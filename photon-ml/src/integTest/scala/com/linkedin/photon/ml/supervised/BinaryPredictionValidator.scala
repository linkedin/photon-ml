package com.linkedin.photon.ml.supervised

import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.supervised.classification.BinaryClassifier
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

/**
 * Verify that on a particular data set, the model only produces finite predictions
 *
 * TODO LOW: think about adding support for other thresholds.
 */
class BinaryPredictionValidator[-GLM <: GeneralizedLinearModel with BinaryClassifier: ClassTag] extends ModelValidator[GLM] {

  override def validateModelPredictions(model:GLM, data:RDD[LabeledPoint]) : Unit = {
    val predictions = model.predictClassAllWithThreshold(data.map(x => x.features) , 0.5)
    val invalidCount = predictions.filter(x => x != BinaryClassifier.negativeClassLabel && x != BinaryClassifier.positiveClassLabel).count
    if (invalidCount > 0) {
      throw new IllegalStateException(s"Found [$invalidCount] samples with invalid predictions (expect [$BinaryClassifier.negativeClassLabel] or [$BinaryClassifier.positiveClassLabel]")
    }
  }
}
