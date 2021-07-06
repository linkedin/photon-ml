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
package com.linkedin.photon.ml

import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.function.DistributedObjectiveFunction
import com.linkedin.photon.ml.function.glm.{LogisticLossFunction, PoissonLossFunction, SquaredLossFunction}
import com.linkedin.photon.ml.normalization.NormalizationContext
import com.linkedin.photon.ml.optimization.OptimizerType.OptimizerType
import com.linkedin.photon.ml.optimization._
import com.linkedin.photon.ml.optimization.game.FixedEffectOptimizationConfiguration
import com.linkedin.photon.ml.supervised.classification.LogisticRegressionModel
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import com.linkedin.photon.ml.supervised.regression.{LinearRegressionModel, PoissonRegressionModel}
import com.linkedin.photon.ml.util.{Logging, PhotonBroadcast}

/**
 * Collection of functions for model training.
 */
object ModelTraining extends Logging {

  /**
   * Train a generalized linear model using the given training dataset and the Photon-ML's parameter settings.
   *
   * @param trainingData The training data represented as a RDD of [[data.LabeledPoint]]
   * @param taskType Learning task type, e.g., LINEAR_REGRESSION or LOGISTIC_REGRESSION or POISSON_REGRESSION
   * @param optimizerType The type of optimizer that will be used to train the model
   * @param regularizationContext The type of regularization that will be used to train the model
   * @param regularizationWeights An array of regularization weights used to train the model
   * @param normalizationContext Normalization context for feature normalization
   * @param maxNumIter Maximum number of iterations to run
   * @param tolerance The optimizer's convergence tolerance, smaller value will lead to higher accuracy with the cost
   *                  of more iterations
   * @param constraintMap An optional mapping of feature indices to box constraints
   * @param treeAggregateDepth The depth for tree aggregation
   * @return The trained models in the form of Map(key -> model), where key is the String typed corresponding
   *         regularization weight used to train the model
   */
  def trainGeneralizedLinearModel(
      trainingData: RDD[LabeledPoint],
      taskType: TaskType.TaskType,
      optimizerType: OptimizerType,
      regularizationContext: RegularizationContext,
      regularizationWeights: List[Double],
      normalizationContext: NormalizationContext,
      maxNumIter: Int,
      tolerance: Double,
      constraintMap: Option[Map[Int, (Double, Double)]],
      treeAggregateDepth: Int,
      useWarmStart: Boolean): List[(Double, GeneralizedLinearModel, OptimizationStatesTracker)] =
    trainGeneralizedLinearModel(
      trainingData,
      taskType,
      optimizerType,
      regularizationContext,
      regularizationWeights,
      normalizationContext,
      maxNumIter,
      tolerance,
      constraintMap,
      Map.empty,
      treeAggregateDepth,
      useWarmStart)

  /**
   * Train a generalized linear model using the given training dataset and the Photon-ML's parameter settings.
   * Sets up a GLM of the appropriate kind then trains it for various regularization weights, performing hyperparameter
   * tuning.
   *
   * @param trainingData The training data represented as a RDD of [[data.LabeledPoint]]
   * @param taskType Learning task type, e.g., LINEAR_REGRESSION or LOGISTIC_REGRESSION or POISSON_REGRESSION
   * @param optimizerType The type of optimizer that will be used to train the model
   * @param regularizationContext The type of regularization that will be used to train the model
   * @param regularizationWeights An array of regularization weights used to train the model
   * @param normalizationContext Normalization context for feature normalization
   * @param maxNumIter Maximum number of iterations to run
   * @param tolerance The optimizer's convergence tolerance, smaller value will lead to higher accuracy with the cost
   *                  of more iterations
   * @param constraintMap An optional mapping of feature indices to box constraints
   * @param warmStartModels Map of (lambda -> model) to use for warm start training
   * @param treeAggregateDepth The depth for tree aggregation
   * @param useWarmStart Whether to use warm start or not in hyperparameter tuning
   * @return The trained models in the form of Map(key -> model), where key is the String typed corresponding
   *         regularization weight used to train the model
   */
  def trainGeneralizedLinearModel(
      trainingData: RDD[LabeledPoint],
      taskType: TaskType.TaskType,
      optimizerType: OptimizerType,
      regularizationContext: RegularizationContext,
      regularizationWeights: List[Double],
      normalizationContext: NormalizationContext,
      maxNumIter: Int,
      tolerance: Double,
      constraintMap: Option[Map[Int, (Double, Double)]],
      warmStartModels: Map[Double, GeneralizedLinearModel],
      treeAggregateDepth: Int,
      useWarmStart: Boolean): List[(Double, GeneralizedLinearModel, OptimizationStatesTracker)] = {

    val optimizerConfig = OptimizerConfig(optimizerType, maxNumIter, tolerance, constraintMap)
    val optimizationConfig = FixedEffectOptimizationConfiguration(optimizerConfig, regularizationContext)
    // Initialize the broadcast normalization context
    val broadcastNormalizationContext = trainingData.sparkContext.broadcast(normalizationContext)
    val wrappedBroadcastNormalizationContext = PhotonBroadcast(broadcastNormalizationContext)

    // Construct the generalized linear optimization problem
    val (glmConstructor, objectiveFunction) = taskType match {
      case TaskType.LOGISTIC_REGRESSION =>
        val constructor = LogisticRegressionModel.apply _
        val objective = DistributedObjectiveFunction(
          optimizationConfig,
          LogisticLossFunction,
          treeAggregateDepth)

        (constructor, objective)

      case TaskType.LINEAR_REGRESSION =>
        val constructor = LinearRegressionModel.apply _
        val objective = DistributedObjectiveFunction(
          optimizationConfig,
          SquaredLossFunction,
          treeAggregateDepth)

        (constructor, objective)

      case TaskType.POISSON_REGRESSION =>
        val constructor = PoissonRegressionModel.apply _
        val objective = DistributedObjectiveFunction(
          optimizationConfig,
          PoissonLossFunction,
          treeAggregateDepth)

        (constructor, objective)

      case _ => throw new Exception(s"Loss function for taskType $taskType is currently not supported.")
    }
    val optimizationProblem = DistributedOptimizationProblem(
      optimizationConfig,
      objectiveFunction,
      samplerOption = None,
      glmConstructor,
      wrappedBroadcastNormalizationContext,
      VarianceComputationType.NONE)

    // Sort the regularization weights from high to low, which would potentially speed up the overall convergence time
    val sortedRegularizationWeights = regularizationWeights.sortWith(_ >= _)

    if (trainingData.getStorageLevel == StorageLevel.NONE) {
      logger.warn("The input data is not directly cached, which may hurt performance if its parent RDDs are also uncached.")
    }

    val numWarmStartModels = warmStartModels.size
    logger.info(s"Starting model fits with $numWarmStartModels warm start models for lambdas " +
      s"${warmStartModels.keys.mkString(", ")}")

    // Hyperparameter tuning
    val initWeightsAndModels = List[(Double, GeneralizedLinearModel, OptimizationStatesTracker)]()
    val finalWeightsModelsAndTrackers = sortedRegularizationWeights
      .foldLeft(initWeightsAndModels) {
        case (List(), currentWeight) =>

          // Initialize the list with the result from the first regularization weight
          optimizationProblem.updateRegularizationWeight(currentWeight)

          val (glm, stateTracker) = if (numWarmStartModels == 0) {

            logger.info(s"No warm start model found; beginning training with a 0-coefficients model")

            optimizationProblem.run(trainingData)

          } else {

            val maxLambda = warmStartModels.keys.max

            logger.info(s"Starting training using warm-start model with lambda = $maxLambda")

            optimizationProblem.run(trainingData, warmStartModels(maxLambda))
          }

          List((currentWeight, glm, stateTracker))

        case (latestWeightsModelsAndTrackers, currentWeight) =>

          optimizationProblem.updateRegularizationWeight(currentWeight)

          // Train the rest of the models
          val (glm, stateTracker) = if (useWarmStart) {
            val previousModel = latestWeightsModelsAndTrackers.head._2

            logger.info(s"Training model with regularization weight $currentWeight started (warm start)")

            optimizationProblem.run(trainingData, previousModel)

          } else {
            logger.info(s"Training model with regularization weight $currentWeight started (no warm start)")

            optimizationProblem.run(trainingData)
          }

          (currentWeight, glm, stateTracker) +: latestWeightsModelsAndTrackers
      }

    broadcastNormalizationContext.unpersist()

    finalWeightsModelsAndTrackers
  }
}
