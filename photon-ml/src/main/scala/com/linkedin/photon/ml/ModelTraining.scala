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
package com.linkedin.photon.ml

import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.normalization.NormalizationContext
import com.linkedin.photon.ml.optimization.OptimizerType.OptimizerType
import com.linkedin.photon.ml.optimization._
import com.linkedin.photon.ml.supervised.TaskType._
import com.linkedin.photon.ml.supervised.classification.{
  LogisticRegressionAlgorithm,
  SmoothedHingeLossLinearSVMAlgorithm}
import com.linkedin.photon.ml.supervised.model.{GeneralizedLinearModel, ModelTracker}
import com.linkedin.photon.ml.supervised.regression.{LinearRegressionAlgorithm, PoissonRegressionAlgorithm}
import org.apache.spark.rdd.RDD

/**
 * Collection of functions for model training
 */
object ModelTraining {

  /**
   * Train a generalized linear model using the given training data set and the Photon-ML's parameter settings
   * @param trainingData The training data represented as a RDD of [[data.LabeledPoint]]
   * @param taskType Learning task type, e.g., LINEAR_REGRESSION or LOGISTIC_REGRESSION or POISSON_REGRESSION
   * @param optimizerType The type of optimizer that will be used to train the model
   * @param regularizationContext The type of regularization that will be used to train the model
   * @param regularizationWeights An array of regularization weights used to train the model
   * @param normalizationContext Normalization context for feature normalization
   * @param maxNumIter Maximum number of iterations to run
   * @param tolerance The optimizer's convergence tolerance, smaller value will lead to higher accuracy with the cost of
   *                  more iterations
   * @param enableOptimizationStateTracker Whether to enable the optimization state tracker, which stores the
   *                                       per-iteration log information of the running optimizer
   * @return The trained models in the form of Map(key -> model), where key is the String typed corresponding
   *   regularization weight used to train the model
   */
  def trainGeneralizedLinearModel(
      trainingData: RDD[LabeledPoint],
      taskType: TaskType,
      optimizerType: OptimizerType,
      regularizationContext: RegularizationContext,
      regularizationWeights: List[Double],
      normalizationContext: NormalizationContext,
      maxNumIter: Int,
      tolerance: Double,
      enableOptimizationStateTracker: Boolean,
      constraintMap: Option[Map[Int, (Double, Double)]],
      treeAggregateDepth: Int): (List[(Double, _ <: GeneralizedLinearModel)], Option[List[(Double, ModelTracker)]]) = {

    trainGeneralizedLinearModel(
      trainingData,
      taskType,
      optimizerType,
      regularizationContext,
      regularizationWeights,
      normalizationContext,
      maxNumIter,
      tolerance,
      enableOptimizationStateTracker,
      constraintMap,
      Map.empty,
      treeAggregateDepth)
  }

  /**
   * Train a generalized linear model using the given training data set and the Photon-ML's parameter settings
   * @param trainingData The training data represented as a RDD of [[data.LabeledPoint]]
   * @param taskType Learning task type, e.g., LINEAR_REGRESSION or LOGISTIC_REGRESSION or POISSON_REGRESSION
   * @param optimizerType The type of optimizer that will be used to train the model
   * @param regularizationContext The type of regularization that will be used to train the model
   * @param regularizationWeights An array of regularization weights used to train the model
   * @param normalizationContext Normalization context for feature normalization
   * @param maxNumIter Maximum number of iterations to run
   * @param tolerance The optimizer's convergence tolerance, smaller value will lead to higher accuracy with the cost of
   *                  more iterations
   * @param enableOptimizationStateTracker Whether to enable the optimization state tracker, which stores the
   *                                       per-iteration log information of the running optimizer
   * @param warmStartModels Map of &lambda; &rarr; model to use for warm start
   * @return The trained models in the form of Map(key -> model), where key is the String typed corresponding
   *   regularization weight used to train the model
   */
  def trainGeneralizedLinearModel(
      trainingData: RDD[LabeledPoint],
      taskType: TaskType,
      optimizerType: OptimizerType,
      regularizationContext: RegularizationContext,
      regularizationWeights: List[Double],
      normalizationContext: NormalizationContext,
      maxNumIter: Int,
      tolerance: Double,
      enableOptimizationStateTracker: Boolean,
      constraintMap: Option[Map[Int, (Double, Double)]],
      warmStartModels:Map[Double, GeneralizedLinearModel],
      treeAggregateDepth: Int): (List[(Double, _ <: GeneralizedLinearModel)], Option[List[(Double, ModelTracker)]]) = {

    val optimizerConfig = OptimizerConfig(optimizerType, maxNumIter, tolerance, constraintMap)

    // Choose the generalized linear algorithm
    val algorithm = taskType match {
      case LINEAR_REGRESSION => new LinearRegressionAlgorithm
      case POISSON_REGRESSION => new PoissonRegressionAlgorithm
      case LOGISTIC_REGRESSION => new LogisticRegressionAlgorithm
      case SMOOTHED_HINGE_LOSS_LINEAR_SVM => new SmoothedHingeLossLinearSVMAlgorithm

      case _ => throw new IllegalArgumentException(s"unrecognized task type $taskType")
    }
    algorithm.isTrackingState = enableOptimizationStateTracker
    algorithm.treeAggregateDepth = treeAggregateDepth

    // Sort the regularization weights from high to low, which would potentially speed up the overall convergence time
    val sortedRegularizationWeights = regularizationWeights.sortWith(_ >= _)

    // Model training with the chosen optimizer and algorithm
    val (models, _) = algorithm.run(
      trainingData,
      optimizerConfig,
      regularizationContext,
      sortedRegularizationWeights,
      normalizationContext,
      warmStartModels)

    val weightModelTuples = sortedRegularizationWeights.zip(models)

    val modelTrackersMapOption = algorithm.getStateTracker
      .map(modelTrackers => sortedRegularizationWeights.zip(modelTrackers))

    (weightModelTuples, modelTrackersMapOption)
  }
}
