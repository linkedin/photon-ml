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
package com.linkedin.photon.ml.supervised.model

import breeze.linalg.Vector
import com.linkedin.photon.ml.data._
import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.function.DiffFunction
import com.linkedin.photon.ml.normalization._
import com.linkedin.photon.ml.optimization.{Optimizer, OptimizerConfig, RegularizationContext}
import org.apache.spark.Logging
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import scala.collection.mutable
import scala.reflect.ClassTag

/**
 * GeneralizedLinearAlgorithm implements methods to train a Generalized Linear Model (GLM).
 * This class should be extended with a loss function and the createModel function to create a new GLM.
 *
 * @tparam GLM The type of returned generalized linear model
 * @tparam Function The type of loss function of the generalized linear algorithm
 */
abstract class GeneralizedLinearAlgorithm[GLM <: GeneralizedLinearModel : ClassTag,
    Function <: DiffFunction[LabeledPoint]]
  extends Logging
  with Serializable {

  /**
   * Optimization state trackers
   */
  protected val modelTrackerBuilder = new mutable.ListBuffer[ModelTracker]()

  /**
   * Whether to track the optimization state (for validating and debugging purpose). Default: True.
   */
  var isTrackingState: Boolean = true

  /**
   * The depth for treeAggregate. Depth 1 indicates normal linear aggregate.
   */
  var treeAggregateDepth: Int = 1

  /**
   * The target storage level if the input data get normalized.
   */
  var targetStorageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK

  /**
   * Get the optimization state trackers for the optimization problems solved in the generalized linear algorithm
   *
   * @return Some(stateTracker) if isTrackingState is set to true, None otherwise.
   */
  def getStateTracker: Option[List[ModelTracker]] = {
    if (isTrackingState) {
      Some(modelTrackerBuilder.toList)
    } else {
      None
    }
  }

  /**
   * Create the objective function of the generalized linear algorithm
   *
   * @param normalizationContext The normalization context for the training
   * @param regularizationContext The type of regularization to construct the objective function
   * @param regularizationWeight The weight of the regularization term in the objective function
   */
  // TODO: enable feature specific regularization / disable regularizing intercept
  protected def createObjectiveFunction(
    normalizationContext: ObjectProvider[NormalizationContext],
    regularizationContext: RegularizationContext,
    regularizationWeight: Double): Function

  /**
   * Create an Optimizer according to the config.
   *
   * @param optimizerConfig Optimizer configuration
   * @return A new Optimizer created according to the configuration
   */
  protected def createOptimizer(optimizerConfig: OptimizerConfig): Optimizer[LabeledPoint, Function]

  /**
   * Create a model given the coefficients
   *
   * @param coefficients The coefficients parameter of each feature (and potentially including intercept)
   * @return A generalized linear model with coefficients parameters
   */
  protected def createModel(coefficients: Vector[Double]): GLM

  /**
   * Create a model given the coefficients
   *
   * @param normalizationContext The normalization context
   * @param coefficients A vector of feature coefficients (and potentially including intercept)
   * @return A generalized linear model with intercept and coefficients parameters
   */
  protected def createModel(normalizationContext: NormalizationContext, coefficients: Vector[Double]): GLM = {
    createModel(normalizationContext.transformModelCoefficients(coefficients))
  }

  /**
   * Run the algorithm with the configured parameters on an input RDD of LabeledPoint entries.
   *
   * @param input A RDD of input labeled data points in the original scale
   * @param optimizerConfig The optimizer config used to construct the optimizer
   * @param regularizationContext The chosen type of regularization
   * @param regularizationWeights An array of weights for the regularization term
   * @param normalizationContext The normalization context
   * @return The learned generalized linear models of each regularization weight and iteration.
   */
  def run(
      input: RDD[LabeledPoint],
      optimizerConfig: OptimizerConfig,
      regularizationContext: RegularizationContext,
      regularizationWeights: List[Double],
      normalizationContext: NormalizationContext): (List[GLM], Optimizer[LabeledPoint, Function]) = {

    logInfo("Doing training without any warm start models")
    run(input, optimizerConfig, regularizationContext, regularizationWeights, normalizationContext, Map.empty)
  }

  /**
   * Run the algorithm with the configured parameters on an input RDD of LabeledPoint entries.
   *
   * @param input A RDD of input labeled data points in the original scale
   * @param optimizerConfig The optimizer config used to construct the optimizer
   * @param regularizationContext The chosen type of regularization
   * @param regularizationWeights An array of weights for the regularization term
   * @param normalizationContext The normalization context
   * @param warmStartModels Map of &lambda; &rarr; suggested warm start.
   * @return The learned generalized linear models of each regularization weight and iteration.
   */
  def run(
      input: RDD[LabeledPoint],
      optimizerConfig: OptimizerConfig,
      regularizationContext: RegularizationContext,
      regularizationWeights: List[Double],
      normalizationContext: NormalizationContext,
      warmStartModels: Map[Double, GeneralizedLinearModel]): (List[GLM], Optimizer[LabeledPoint, Function]) = {

    val numFeatures = input.first().features.size
    val initialWeight = Vector.zeros[Double](numFeatures)
    val initialModel = createModel(initialWeight)

    run(input,
      initialModel,
      optimizerConfig,
      regularizationContext,
      regularizationWeights,
      normalizationContext,
      warmStartModels)
  }

  /**
   * Run the algorithm with the configured parameters on an input RDD of LabeledPoint entries
   * starting from the initial model provided.
   *
   * @param input A RDD of input labeled data points in the normalized scale (if normalization is enabled)
   * @param initialModel The initial model
   * @param optimizerConfig The optimizer config used to construct the optimizer
   * @param regularizationContext The chosen type of regularization
   * @param regularizationWeights An array of weights for the regularization term
   * @param normalizationContext The normalization context
   * @param warmStartModels Optional suggested models for warm start
   * @return The learned generalized linear models of each regularization weight and iteration.
   */
  protected def run(
      input: RDD[LabeledPoint],
      initialModel: GLM,
      optimizerConfig: OptimizerConfig,
      regularizationContext: RegularizationContext,
      regularizationWeights: List[Double],
      normalizationContext: NormalizationContext,
      warmStartModels: Map[Double, GeneralizedLinearModel]): (List[GLM], Optimizer[LabeledPoint, Function]) = {

    logInfo(s"Starting model fits with ${warmStartModels.size} warm start models for " +
            s"lambda = ${warmStartModels.keys.mkString(", ")}")

    if (input.getStorageLevel == StorageLevel.NONE) {
      logWarning("The input data is not directly cached, which may hurt performance if its"
        + " parent RDDs are also uncached.")
    }

    val broadcastNormalizationContext = input.sparkContext.broadcast(normalizationContext)
    val normalizationContextProvider =
      new BroadcastedObjectProvider[NormalizationContext](broadcastNormalizationContext)

    val largestWarmStartLambda = if (warmStartModels.isEmpty) {
      0.0
    } else {
      warmStartModels.keys.max
    }
    if (warmStartModels.nonEmpty) {
      logInfo(s"Starting training using warm-start model with lambda = $largestWarmStartLambda")
    } else {
      logInfo(s"No warm start model found; falling back to all 0 as initial value")
    }

    //
    // Find the path of solutions with different regularization coefficients
    //

    val optimizer = createOptimizer(optimizerConfig)
    optimizer.setStateTrackingEnabled(isTrackingState)
    // Reuse the previous initial state for consistent convergence checks over consecutive runs in the grid-search-based
    // hyper-parameter tuning procedure.
    optimizer.isReusingPreviousInitialState = true

    // If we can find a warm start model for the largest lambda, use that. Otherwise, default to the provided initial
    // model.
    val initModel = warmStartModels.getOrElse(largestWarmStartLambda, initialModel)
    var initialCoefficients = initModel.coefficients

    val models = regularizationWeights.map { regularizationWeight =>
      val objectiveFunction = createObjectiveFunction(
        normalizationContextProvider,
        regularizationContext,
        regularizationWeight)
      val (optimizedCoefficients, _) = optimizer.optimize(input, objectiveFunction, initialCoefficients)

      initialCoefficients = optimizedCoefficients
      logInfo(s"Training model with regularization weight $regularizationWeight finished")

      if (isTrackingState) {
        val tracker = optimizer.getStateTracker.get
        logInfo(s"History tracker information:\n $tracker")
        val modelsPerIteration = tracker.getTrackedStates.map(x => createModel(normalizationContext, x.coefficients))
        modelTrackerBuilder += new ModelTracker(tracker.toString, modelsPerIteration)
        logInfo(s"Number of iterations: ${modelsPerIteration.length}")
      }

      createModel(normalizationContext, optimizedCoefficients)
    }
    broadcastNormalizationContext.unpersist()

    (models, optimizer)
  }
}
