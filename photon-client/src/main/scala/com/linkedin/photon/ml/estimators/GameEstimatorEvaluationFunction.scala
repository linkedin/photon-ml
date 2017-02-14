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
package com.linkedin.photon.ml.estimators

import scala.collection.SortedMap
import scala.collection.mutable.Queue

import breeze.linalg.DenseVector
import org.apache.spark.sql.DataFrame

import com.linkedin.photon.ml.estimators.GameEstimator.GameResult
import com.linkedin.photon.ml.hyperparameter.EvaluationFunction
import com.linkedin.photon.ml.normalization.NormalizationContext
import com.linkedin.photon.ml.optimization.game.GameModelOptimizationConfiguration

/**
 * Evaluation function implementation for GAME
 *
 * An evaluation function is the integration point between the hyperparameter tuning module and an estimator, or any
 * system that can unpack a vector of values and produce a real evaluation.
 */
class GameEstimatorEvaluationFunction(
    estimator: GameEstimator,
    optimizationConfiguration: GameModelOptimizationConfiguration,
    data: DataFrame,
    validationData: DataFrame)
  extends EvaluationFunction[GameResult] {

  /**
   * Performs the evaluation
   *
   * @param hyperParameters the vector of hyperparameter values under which to evaluate the function
   * @return a tuple of the evaluated value and the original output from the inner estimator
   */
  override def apply(hyperParameters: DenseVector[Double]): (Double, GameResult) = {
    val newConfiguration = vectorToConfiguration(hyperParameters)

    val model = estimator.fit(data, Some(validationData), Seq(newConfiguration)).head
    val (_, Some(evaluations), _) = model

    // Assumes model selection evaluator is in "head" position
    (evaluations.head._2, model)
  }

  /**
   * Extracts a vector representation from the hyperparameters associated with the original estimator output
   *
   * @param gameResult the original estimator output
   * @return vector representation
   */
  override def vectorizeParams(gameResult: GameResult): DenseVector[Double] =
    configurationToVector(gameResult._3)

  /**
   * Extracts the evaluated value from the original estimator output
   *
   * @param gameResult the original estimator output
   * @return the evaluated value
   */
  override def getEvaluationValue(gameResult: GameResult): Double = gameResult match {
    case (_, Some(evaluations), _) =>
      // We assume the model selection evaluator is in head position
      evaluations.head._2

    case _ => throw new IllegalArgumentException(
      s"Can't extract evaluation value from a GAME result with no evaluations: $gameResult")
  }

  /**
   * Computes the number of hyperparameters from the model optimization configuration
   *
   * @param configuration the optimization configuration from one iteration of GAME training
   * @return the number of hyperparameters
   */
  protected[ml] def numParams(configuration: GameModelOptimizationConfiguration): Int =
    configurationToVector(configuration).length

  /**
   * Extracts a vector representation from the hyperparameters associated with the original estimator output
   *
   * @param configuration the GAME optimization configuration containing parameters
   * @return vector representation
   */
  protected[ml] def configurationToVector(
      configuration: GameModelOptimizationConfiguration): DenseVector[Double] = {

    // Use sorted maps to ensure consistent vector layout
    val fixedEffectOptimizationConfiguration = SortedMap(configuration.fixedEffectOptimizationConfiguration.toSeq: _*)
    val randomEffectOptimizationConfiguration = SortedMap(configuration.randomEffectOptimizationConfiguration.toSeq: _*)
    val factoredRandomEffectOptimizationConfiguration =
      SortedMap(configuration.factoredRandomEffectOptimizationConfiguration.toSeq: _*)

    // Pack the fixed-effect hyperparameters
    val feVals = fixedEffectOptimizationConfiguration.values.map(_.regularizationWeight)

    // Pack the random effect hyperparameters
    val reVals = randomEffectOptimizationConfiguration.values.map(_.regularizationWeight)

    // Pack the factored random effect hyperparameters
    val factoredReVals = factoredRandomEffectOptimizationConfiguration.values.flatMap(config =>
        List(config.randomEffectOptimizationConfiguration.regularizationWeight,
          config.latentFactorOptimizationConfiguration.regularizationWeight))

    DenseVector((feVals ++ reVals ++ factoredReVals).toArray)
  }

  /**
   * Unpacks the regularization weights from the hyperparameter vector, and returns an equivalent GAME optimization
   * configuration
   *
   * @param hyperParameters the hyperparameter vector
   * @return the equivalent GAME optimization configuration
   */
  protected[ml] def vectorToConfiguration(
      hyperParameters: DenseVector[Double]): GameModelOptimizationConfiguration = {

    val paramValues = Queue(hyperParameters.toArray: _*)

    // Use sorted maps to ensure that ordering of hyperparamters aligns with the input vector
    val fixedEffectOptimizationConfiguration = SortedMap(
      optimizationConfiguration.fixedEffectOptimizationConfiguration.toSeq: _*)
    val randomEffectOptimizationConfiguration = SortedMap(
      optimizationConfiguration.randomEffectOptimizationConfiguration.toSeq: _*)
    val factoredRandomEffectOptimizationConfiguration = SortedMap(
      optimizationConfiguration.factoredRandomEffectOptimizationConfiguration.toSeq: _*)

    val n = numParams(optimizationConfiguration)

    require(paramValues.length == n,
      "Dimension mismatch between the parameter vector and the actual parameters.")

    optimizationConfiguration.copy(
      // Unpack the fixed-effect hyperparameters
      fixedEffectOptimizationConfiguration =
        fixedEffectOptimizationConfiguration.mapValues { config =>
          config.copy(regularizationWeight = paramValues.dequeue)
        }.view.force.toMap,

      // Unpack the random effect hyperparameters
      randomEffectOptimizationConfiguration =
        randomEffectOptimizationConfiguration.mapValues { config =>
          config.copy(regularizationWeight = paramValues.dequeue)
        }.view.force.toMap,

      // Unpack the factored random effect hyperparameters
      factoredRandomEffectOptimizationConfiguration =
       factoredRandomEffectOptimizationConfiguration.mapValues { config =>
         config.copy(
           randomEffectOptimizationConfiguration =
             config.randomEffectOptimizationConfiguration.copy(regularizationWeight = paramValues.dequeue),
           latentFactorOptimizationConfiguration =
             config.latentFactorOptimizationConfiguration.copy(regularizationWeight = paramValues.dequeue))
         }.view.force.toMap
    )
  }
}
