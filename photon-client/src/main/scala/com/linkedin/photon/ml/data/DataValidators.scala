/*
 * Copyright 2017 LinkedIn Corp. All rights reserved.
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
package com.linkedin.photon.ml.data

import breeze.linalg.{DenseVector, SparseVector}
import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.DataValidationType.DataValidationType
import com.linkedin.photon.ml.{DataValidationType, TaskType}
import com.linkedin.photon.ml.TaskType.TaskType
import com.linkedin.photon.ml.supervised.classification.BinaryClassifier
import com.linkedin.photon.ml.util.Logging

/**
 * A collection of methods used to validate data before applying ML algorithms.
 */
object DataValidators extends Logging {
  val linearRegressionValidator: RDD[LabeledPoint] => Boolean = { data =>
    validateFeatures(
      data, Map(
        "Finite labels" -> finiteLabel,
        "Finite features" -> finiteFeatures,
        "Finite offsets" -> finiteOffset))
  }

  val logisticRegressionValidator: RDD[LabeledPoint] => Boolean = { data =>
    validateFeatures(
      data, Map(
        "Binary labels" -> binaryLabel,
        "Finite features" -> finiteFeatures,
        "Finite offsets" -> finiteOffset))
  }

  val poissonRegressionValidator: RDD[LabeledPoint] => Boolean = { data =>
    validateFeatures(
      data, Map(
        "Finite labels" -> finiteLabel,
        "Non-negative labels" -> nonNegativeLabels,
        "Finite features" -> finiteFeatures,
        "Finite offsets" -> finiteOffset))
  }

  /**
   *
   * @param labeledPoint
   * @return
   */
  def nonNegativeLabels(labeledPoint: LabeledPoint): Boolean = {
    labeledPoint.label >= 0
  }

  /**
   *
   * @param labeledPoint
   * @return
   */
  def finiteLabel(labeledPoint: LabeledPoint): Boolean = {
    !(labeledPoint.label.isNaN || labeledPoint.label.isInfinite)
  }

  /**
   *
   * @param labeledPoint
   * @return
   */
  def finiteOffset(labeledPoint: LabeledPoint): Boolean = {
    !(labeledPoint.offset.isNaN || labeledPoint.offset.isInfinite)
  }

  /**
   *
   * @param labeledPoint
   * @return
   */
  def binaryLabel(labeledPoint: LabeledPoint): Boolean = {
    BinaryClassifier.positiveClassLabel == labeledPoint.label ||
      BinaryClassifier.negativeClassLabel == labeledPoint.label
  }

  /**
   *
   * @param labeledPoint
   * @return
   */
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

  /**
   *
   * @param dataSet
   * @param perSampleValidators
   * @return
   */
  def validateFeatures(dataSet:RDD[LabeledPoint], perSampleValidators:Map[String, LabeledPoint=>Boolean]): Boolean = {
    dataSet.mapPartitions(x => {
      Seq(x.forall(item => {
        perSampleValidators.map( validator => {
          val valid = validator._2(item)
          if (!valid) {
            logger.error(s"Validation ${validator._1} failed on item: $item")
          }
          valid
        }).forall(x => x)
      })).iterator
    }).fold(true)(_ && _)
  }

  /**
   *
   * @param inputData
   * @param taskType
   * @param dataValidationType
   * @return
   */
  def sanityCheckData(
      inputData: RDD[LabeledPoint],
      taskType: TaskType,
      dataValidationType: DataValidationType): Boolean = {

    if (! dataValidationType.equals(DataValidationType.VALIDATE_DISABLED)) {
      // Check the data properties
      val validators: Seq[RDD[LabeledPoint] => Boolean] = taskType match {
        case TaskType.LINEAR_REGRESSION => List(DataValidators.linearRegressionValidator)
        case TaskType.LOGISTIC_REGRESSION => List(DataValidators.logisticRegressionValidator)
        case TaskType.POISSON_REGRESSION => List(DataValidators.poissonRegressionValidator)
        case TaskType.SMOOTHED_HINGE_LOSS_LINEAR_SVM => List(DataValidators.logisticRegressionValidator)
      }
      dataValidationType match {
        case DataValidationType.VALIDATE_FULL =>
          val valid = validators.map(x => x(inputData)).forall(x => x)
          if (valid) {
            true
          } else {
            logger.error("Data validation failed.")
            false
          }
        case DataValidationType.VALIDATE_SAMPLE =>
          logger.warn("Doing a partial validation on ~10% of the training data")
          val subset = inputData.sample(withReplacement = false, fraction = 0.10)
          val valid = validators.map(x => x(subset)).forall(x => x)
          if (valid) {
            true
          } else {
            logger.error("Data validation failed.")
            false
          }
      }
    } else {
      logger.warn("Data validation disabled.")
      true
    }
  }
}
