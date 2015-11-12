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
package com.linkedin.photon.ml.normalization

import breeze.linalg.{DenseVector, SparseVector}
import com.linkedin.photon.ml.{DataValidationType, ModelTraining}
import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.optimization.{L2RegularizationContext, OptimizerType}
import com.linkedin.photon.ml.stat.BasicStatistics
import com.linkedin.photon.ml.supervised.TaskType
import com.linkedin.photon.ml.supervised.classification.{BinaryClassifier, LogisticRegressionModel}
import com.linkedin.photon.ml.test.SparkTestUtils
import org.apache.spark.rdd.RDD
import org.testng.Assert._
import org.testng.annotations.Test

import scala.util.Random


/**
 * All feature normalizations are affine transformation so the resulting models without regularization should be the same
 * except for numerical errors.
 *
 * This test checks the validity of feature normalization in the context of training.
 *
 * @author dpeng
 */
class NormalizationIntegTest extends SparkTestUtils {
  private val _seed = 1
  private val _size = 100
  private val _dimension = 10
  private val _threshold = 0.5
  private val _tolerance = 1.0E-5
  private val _numIter = 100
  private val _precision = 0.95

  @Test
  def testNormalization(): Unit = sparkTest("testNormalization") {
    val model = generateRandomModel(_seed)

    val trainRDD = generateSampleRDD(_seed, model)
    val testRDD = generateSampleRDD(_seed + 1, model)
    NormalizationType.values().foreach(checkTrainingOfNormalizationType(trainRDD, testRDD, _))
  }

  /**
   * Generate a random model using a random dense vector
   * @param seed The random seed used to generate the model
   * @return A [[LogisticRegressionModel]]
   */
  private def generateRandomModel(seed: Int): LogisticRegressionModel = {
    Random.setSeed(seed)
    // The size of the vector is _dimension + 1 due to the intercept
    val coef = (for (i <- 0 to _dimension) yield Random.nextGaussian()).toArray
    new LogisticRegressionModel(DenseVector(coef), None)
  }

  /**
   * Generate sample data for binary classification problems according to a random seed and a model. The labels in the data
   * is exactly predicted by the input model.
   * @param seed The random seed used to generate the data
   * @param model The input model used to generate the labels in the data
   * @return The data RDD
   */
  private def generateSampleRDD(seed: Int, model: LogisticRegressionModel): RDD[LabeledPoint] = {
    val data = drawBalancedSampleFromNumericallyBenignDenseFeaturesForBinaryClassifierLocal(seed, _size, _dimension).map {
      case (_, sparseVector: SparseVector[Double]) =>
        val size = sparseVector.size
        // Append data with the intercept
        val vector = new SparseVector[Double](sparseVector.index :+ size,
                                              sparseVector.data :+ 1.0,
                                              size + 1)
        // Replace the random label using the one predicted by the input model
        val label = model.predictClass(vector, _threshold)
        new LabeledPoint(label, vector)
    }.toArray
    sc.parallelize(data)
  }

  /**
   * Check the correctness of training with a specific normalization type. This check involves an unregularized training. The prediction of an unregularized
   * trained model should predictClass exactly the same label as the one in the input training data, given that the labels are generated by a model without noise.
   * This method also checks the precision of a test data set.
   * @param trainRDD Training data set
   * @param testRDD Test data set
   * @param normalizationType Normalization type
   */
  private def checkTrainingOfNormalizationType(trainRDD: RDD[LabeledPoint], testRDD: RDD[LabeledPoint],
                                               normalizationType: NormalizationType): Unit = {
    // This is necessary to make Spark not complain serialization error of this class.
    val summary = BasicStatistics.getBasicStatistics(trainRDD)
    val normalizationContext = NormalizationContext(normalizationType, summary, Some(_dimension))
    val threshold = _threshold
    val (models, _) = ModelTraining.trainGeneralizedLinearModel(
      trainRDD,
      TaskType.LOGISTIC_REGRESSION,
      OptimizerType.LBFGS,
      L2RegularizationContext,
      List(0.0),
      normalizationContext,
      _numIter,
      _tolerance,
      true,
      DataValidationType.VALIDATE_DISABLED,
      None)
    assertEquals(models.size, 1)
    val model = models(0)._2.asInstanceOf[BinaryClassifier]
    // For all types of normalization, the unregularized trained model should predictClass the same label.
    trainRDD.foreach {
      case LabeledPoint(label, vector, _, _) =>
        val prediction = model.predictClass(vector, threshold)
        assertEquals(prediction, label)
    }

    // For a test data set, the trained model should recover a certain level of precision.
    val correct = testRDD.filter {
      case LabeledPoint(label, vector, _, _) =>
        val prediction = model.predictClass(vector, threshold)
        label == prediction
    }.count

    assertTrue(correct.toDouble / _size >= _precision, s"Precision check [${correct} / ${_size} >= ${_precision}] failed.")
  }
}
