/*
 * Copyright 2014 LinkedIn Corp. All rights reserved.
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
package com.linkedin.photon.ml.util

import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.supervised.classification.BinaryClassifier
import org.apache.spark.Logging
import org.apache.spark.rdd.RDD

/**
 * A collection of methods used to validate data before applying ML algorithms.
 */
object DataValidators extends Logging {

  private val epsilon = 1e-15

  /**
   * Function to check if labels used for classification are either zero or one.
   *
   * @return True if labels are all zero or one, false otherwise.
   */
  val binaryLabelValidator: RDD[LabeledPoint] => Boolean = { data =>
    val numInvalid = data.filter(x => math.abs(x.label - BinaryClassifier.positiveClassLabel) > epsilon
      && math.abs(x.label - BinaryClassifier.negativeClassLabel) > epsilon).count()
    if (numInvalid != 0) {
      logError(s"Classification labels should be ${BinaryClassifier.negativeClassLabel.toInt} or" +
        s"${BinaryClassifier.positiveClassLabel.toInt}. Found $numInvalid invalid labels")
    }
    numInvalid == 0
  }

  /**
   * Function to check if labels used for Poisson Regression are non-negative.
   *
   * @return True if labels are all non-negative, false otherwise.
   */
  val nonNegativeLabelValidator: RDD[LabeledPoint] => Boolean = { data =>
    val numInvalid = data.filter(x =>  x.label < 0).count()
    if (numInvalid != 0) {
      logError(s"Labels should be non-negative. Found $numInvalid invalid labels")
    }
    numInvalid == 0
  }

  /**
   * Check that all labels are finite (Double.isFinite)
   */
  val finiteLabelValidator: RDD[LabeledPoint] => Boolean = { data =>
    val numInvalid = data.filter(x => !java.lang.Double.isFinite(x.label)).count()

    if (numInvalid != 0) {
      logError(s"Labels should be finite (_not_ NaN / Inf / -Inf). Found $numInvalid invalid labels")
    }
    numInvalid == 0
  }

  /**
   * Check all present feature values are finite
   */
  val finiteFeaturesValidator: RDD[LabeledPoint] => Boolean = { data =>
    val numInvalid = data.filter( x => {
      val featureCount:Int = x.features.mapActiveValues( y => { if (java.lang.Double.isFinite(y)) 0 else 1 }).sum
      val offsetCount:Int = if (java.lang.Double.isFinite(x.offset)) { 0 } else { 1 }
      val nonFiniteCount = featureCount + offsetCount
      nonFiniteCount > 0
    }).count

    if (numInvalid != 0) {
      logError(s"Feature values should should be finite (_not_ NaN / Inf / -Inf). Found $numInvalid invalid labels")
    }
    numInvalid == 0
  }
}