/*
 * Copyright 2015 LinkedIn Corp. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License"); you may
 * not use this file except in compliance with the License. You may obtain a
 * copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */
package com.linkedin.photon.ml.supervised

import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.supervised.classification.BinaryClassifier
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

/**
 * Verify that we are able to achieve some minimum AUROC as part of validating a binary classifier's predictions
 */
class BinaryClassifierAUCValidator[-BC <: GeneralizedLinearModel with BinaryClassifier : ClassTag](minimumAUC:Double) extends ModelValidator[BC] {
  assert(minimumAUC >= 0.5)
  assert(minimumAUC <= 1.0)

  def validateModelPredictions(model:BC, data:RDD[LabeledPoint]) = {
    val scored:RDD[(Double, Double)] = data.map { x => (x.label, model.computeMeanFunctionWithOffset(x.features, x.offset))}
    val evaluator = new BinaryClassificationMetrics(scored)
    val auROC = evaluator.areaUnderROC

    if (auROC < minimumAUC) {
      throw new IllegalStateException(s"Computed AUROC [$auROC] is smaller than minimum required [$minimumAUC]")
    }
  }
}
