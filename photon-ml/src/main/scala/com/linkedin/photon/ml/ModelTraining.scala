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

import com.linkedin.photon.ml.data.{BroadcastedObjectProvider, LabeledPoint}
import com.linkedin.photon.ml.function.DiffFunction
import com.linkedin.photon.ml.normalization.NormalizationContext
import com.linkedin.photon.ml.optimization.OptimizerType.OptimizerType
import com.linkedin.photon.ml.optimization._
import com.linkedin.photon.ml.optimization.game.GLMOptimizationConfiguration
import com.linkedin.photon.ml.supervised.TaskType._
import com.linkedin.photon.ml.supervised.model.{GeneralizedLinearModel, ModelTracker}
import org.apache.spark.Logging
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

/**
  * Collection of functions for model training
 */
object ModelTraining extends Logging {

  /**
    * Train a generalized linear model using the given training data set and the Photon-ML's parameter settings
    *
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
    *
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
      warmStartModels: Map[Double, GeneralizedLinearModel],
      treeAggregateDepth: Int): (List[(Double, _ <: GeneralizedLinearModel)], Option[List[(Double, ModelTracker)]]) = {

    val optimizerConfig = OptimizerConfig(optimizerType, maxNumIter, tolerance, constraintMap)
    val optimizationConfig = GLMOptimizationConfiguration(optimizerConfig, regularizationContext)

    // Choose the generalized linear algorithm
    val initOptimizationProblem:
        GeneralizedLinearOptimizationProblem[GeneralizedLinearModel, DiffFunction[LabeledPoint]] = taskType match {

      case LOGISTIC_REGRESSION => LogisticRegressionOptimizationProblem.buildOptimizationProblem(
        optimizationConfig,
        treeAggregateDepth,
        enableOptimizationStateTracker)

      case LINEAR_REGRESSION => LinearRegressionOptimizationProblem.buildOptimizationProblem(
        optimizationConfig,
        treeAggregateDepth,
        enableOptimizationStateTracker)

      case POISSON_REGRESSION => PoissonRegressionOptimizationProblem.buildOptimizationProblem(
        optimizationConfig,
        treeAggregateDepth,
        enableOptimizationStateTracker)

      case SMOOTHED_HINGE_LOSS_LINEAR_SVM => SmoothedHingeLossLinearSVMOptimizationProblem.buildOptimizationProblem(
        optimizationConfig,
        treeAggregateDepth,
        enableOptimizationStateTracker)

      case _ => throw new IllegalArgumentException(s"unrecognized task type $taskType")
    }

    // Initialize the broadcasted normalization context
    val broadcastNormalizationContext = trainingData.sparkContext.broadcast(normalizationContext)
    val normalizationContextProvider =
      new BroadcastedObjectProvider[NormalizationContext](broadcastNormalizationContext)

    // Sort the regularization weights from high to low, which would potentially speed up the overall convergence time
    val sortedRegularizationWeights = regularizationWeights.sortWith(_ >= _)

    if (trainingData.getStorageLevel == StorageLevel.NONE) {
      logWarning("The input data is not directly cached, which may hurt performance if its parent RDDs are also uncached.")
    }

    val numWarmStartModels = warmStartModels.size
    logInfo(s"Starting model fits with $numWarmStartModels warm start models for lambdas " +
      s"${warmStartModels.keys.mkString(", ")}")

    val initWeightsAndModels = List[(Double, GeneralizedLinearModel)]()
    val (finalOptimizationProblem, finalWeightsAndModels) = sortedRegularizationWeights
      .foldLeft((initOptimizationProblem, initWeightsAndModels)) {
        case ((optimizationProblem, List()), currentWeight) =>
          // Initialize the list with the result from the first regularization weight
          val updatedOptimizationProblem = optimizationProblem.updateObjective(
            normalizationContextProvider, currentWeight)

          if (numWarmStartModels == 0) {
            logInfo(s"No warm start model found; beginning training with a 0-coefficients model")

            (updatedOptimizationProblem,
              List((currentWeight, updatedOptimizationProblem.run(trainingData, normalizationContext))))
          } else {
            val maxLambda = warmStartModels.keys.max
            logInfo(s"Starting training using warm-start model with lambda = $maxLambda")

            (updatedOptimizationProblem,
              List((currentWeight,
                updatedOptimizationProblem.run(
                  trainingData, warmStartModels.get(maxLambda).get, normalizationContext))))
          }

        case ((latestOptimizationProblem, latestWeightsAndModels), currentWeight) =>
          // Train the rest of the models
          val previousModel = latestWeightsAndModels.head._2
          val updatedOptimizationProblem = latestOptimizationProblem.updateObjective(
            normalizationContextProvider, currentWeight)

          logInfo(s"Training model with regularization weight $currentWeight finished")

          (updatedOptimizationProblem,
            (currentWeight, updatedOptimizationProblem.run(trainingData, previousModel, normalizationContext))
              :: latestWeightsAndModels)
      }

    val finalWeightsAndModelTrackers = finalOptimizationProblem.getModelTracker.map(sortedRegularizationWeights.zip(_))

    (finalWeightsAndModels.reverse, finalWeightsAndModelTrackers)
  }
}
