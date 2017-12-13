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

import scala.collection.mutable
import scala.math.{exp, log}

import breeze.linalg.DenseVector
import org.apache.spark.sql.DataFrame

import com.linkedin.photon.ml.estimators.GameEstimator.{GameOptimizationConfiguration, GameResult}
import com.linkedin.photon.ml.hyperparameter.EvaluationFunction
import com.linkedin.photon.ml.optimization.game.{FixedEffectOptimizationConfiguration, RandomEffectOptimizationConfiguration}

/**
 * Evaluation function implementation for GAME.
 *
 * An evaluation function is the integration point between the hyperparameter tuning module and an estimator, or any
 * system that can unpack a vector of values and produce a real evaluation.
 */
class GameEstimatorEvaluationFunction(
    estimator: GameEstimator,
    baseConfig: GameOptimizationConfiguration,
    data: DataFrame,
    validationData: DataFrame)
  extends EvaluationFunction[GameResult] {

  // CoordinateOptimizationConfigurations sorted in order by coordinate ID name
  private val baseConfigSeq = baseConfig.toSeq.sortBy(_._1)

  // Number of parameters in the base configuration
  val numParams: Int = baseConfigSeq.length

  /**
   * Performs the evaluation.
   *
   * @param hyperParameters The vector of hyperparameter values under which to evaluate the function
   * @return A tuple of the evaluated value and the original output from the inner estimator
   */
  override def apply(hyperParameters: DenseVector[Double]): (Double, GameResult) = {
    val newConfiguration = vectorToConfiguration(hyperParameters)

    val model = estimator.fit(data, Some(validationData), Seq(newConfiguration)).head
    val (_, Some(evaluations), _) = model

    // Assumes model selection evaluator is in "head" position
    (evaluations.head._2, model)
  }

  /**
   * Extracts a vector representation from the hyperparameters associated with the original estimator output.
   *
   * @param gameResult The original estimator output
   * @return A vector representation of hyperparameters for a [[GameResult]]
   */
  override def vectorizeParams(gameResult: GameResult): DenseVector[Double] =
    configurationToVector(gameResult._3)

  /**
   * Extracts the evaluated value from the original estimator output.
   *
   * @param gameResult The original estimator output
   * @return The evaluated value
   */
  override def getEvaluationValue(gameResult: GameResult): Double = gameResult match {
    case (_, Some(evaluations), _) =>
      // We assume the model selection evaluator is in head position
      evaluations.head._2

    case _ => throw new IllegalArgumentException(
      s"Can't extract evaluation value from a GAME result with no evaluations: $gameResult")
  }

  /**
   * Extracts a vector representation from the hyperparameters associated with the original estimator output.
   *
   * @param configuration The GAME optimization configuration containing parameters
   * @return A vector representation of hyperparameters for a [[GameOptimizationConfiguration]]
   */
  protected[ml] def configurationToVector(configuration: GameOptimizationConfiguration): DenseVector[Double] = {

    // Input configurations must contain the exact same coordinates as the base configuration
    require(
      baseConfig.size == configuration.size,
      s"Configuration dimension mismatch; ${baseConfig.size} != ${configuration.size}")
    baseConfig.foreach { case (coordinateId, optConfig) =>
      require(configuration.contains(coordinateId), s"Configuration missing initial coordinate $coordinateId")
      require(
        configuration(coordinateId).getClass == optConfig.getClass,
        s"Configuration has mismatched types for coordinate $coordinateId; " +
          s"${optConfig.getClass} != ${configuration(coordinateId).getClass}")
    }

    val parameterArray = configuration
      .toSeq
      .sortBy(_._1)
      .map { case (_, optConfig) =>
        optConfig match {
          case fixed: FixedEffectOptimizationConfiguration => log(fixed.regularizationWeight)

          case random: RandomEffectOptimizationConfiguration => log(random.regularizationWeight)

          case other =>
            throw new IllegalArgumentException(s"Unknown coordinate optimization configuration type: ${other.getClass}")
        }
      }
      .toArray

    DenseVector(parameterArray)
  }

  /**
   * Unpacks the regularization weights from the hyperparameter vector, and returns an equivalent GAME optimization
   * configuration.
   *
   * @param hyperParameters The hyperparameter vector
   * @return The equivalent GAME optimization configuration
   */
  protected[ml] def vectorToConfiguration(hyperParameters: DenseVector[Double]): GameOptimizationConfiguration = {

    require(
      hyperParameters.length == numParams,
      s"Configuration dimension mismatch; $numParams != ${hyperParameters.length}")

    val paramValues = mutable.Queue(hyperParameters.toArray: _*)

    baseConfigSeq
      .map { case (coordinateId, coordinateConfig) =>
        val newCoordinateConfig = coordinateConfig match {
          case fixed: FixedEffectOptimizationConfiguration =>
            fixed.copy(regularizationWeight = exp(paramValues.dequeue()))

          case random: RandomEffectOptimizationConfiguration =>
            random.copy(regularizationWeight = exp(paramValues.dequeue()))

          case other =>
            throw new IllegalArgumentException(s"Unknown coordinate optimization configuration type: ${other.getClass}")
        }

        (coordinateId, newCoordinateConfig)
      }
      .toMap
  }
}
