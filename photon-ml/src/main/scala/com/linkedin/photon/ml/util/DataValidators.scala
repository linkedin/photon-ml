/*
 * Copyright 2016 LinkedIn Corp. All rights reserved.
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

import breeze.linalg.{DenseVector, SparseVector}
import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.supervised.classification.BinaryClassifier
import org.apache.spark.Logging
import org.apache.spark.rdd.RDD

/**
 * A collection of methods used to validate data before applying ML algorithms.
 */
object DataValidators extends Logging {
  val linearRegressionValidator: RDD[LabeledPoint] => Boolean = { data =>
    validateFeatures(data, Map("Finite features" -> finiteFeatures, "Finite labels" -> finiteLabel, "Finite offset" -> finiteOffset))
  }

  val logisticRegressionValidator: RDD[LabeledPoint] => Boolean = { data =>
    validateFeatures(data, Map("Finite label" -> finiteLabel, "Binary label" -> binaryLabel, "Finite features" -> finiteFeatures, "Finite offset" -> finiteOffset))
  }

  val poissonRegressionValidator: RDD[LabeledPoint] => Boolean = { data =>
    validateFeatures(data, Map("Finite label" -> finiteLabel, "Nonnegative label" -> nonnegativeLabels, "Finite features" -> finiteFeatures, "Finite offset" -> finiteOffset))
  }

  def nonnegativeLabels(labeledPoint: LabeledPoint): Boolean = {
    labeledPoint.label >= 0
  }

  def finiteLabel(labeledPoint: LabeledPoint): Boolean = {
    !(labeledPoint.label.isNaN || labeledPoint.label.isInfinite)
  }

  def finiteOffset(labeledPoint: LabeledPoint): Boolean = {
    !(labeledPoint.offset.isNaN || labeledPoint.offset.isInfinite)
  }

  def binaryLabel(labeledPoint: LabeledPoint): Boolean = {
    BinaryClassifier.positiveClassLabel == labeledPoint.label || BinaryClassifier.negativeClassLabel == labeledPoint.label
  }

  def finiteFeatures(labeledPoint: LabeledPoint): Boolean = {
    var result = true
    var idx = 0
    while (idx < labeledPoint.features.activeSize && result) {
      val value = labeledPoint.features match {
        case d:DenseVector[Double] => d.valueAt(idx)
        case s:SparseVector[Double] => s.valueAt(idx)
      }
      result = result && !(value.isNaN || value.isInfinite)
      idx += 1
    }

    result
  }

  def validateFeatures(dataSet:RDD[LabeledPoint], perSampleValidators:Map[String, LabeledPoint=>Boolean]): Boolean = {
    dataSet.mapPartitions(x => {
      Seq(x.forall(item => {
        perSampleValidators.map( validator => {
          val valid = validator._2(item)
          if (!valid) {
            logError(s"Validation ${validator._1} failed on item: ${item}")
          }
          valid
        }).forall(x => x)
      })).iterator
    }).fold(true)(_ && _)
  }
}