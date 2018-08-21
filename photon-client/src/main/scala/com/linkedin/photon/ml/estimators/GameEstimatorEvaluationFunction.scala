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
import com.linkedin.photon.ml.hyperparameter.{EvaluationFunction, VectorRescaling}
import com.linkedin.photon.ml.optimization.{ElasticNetRegularizationContext, RegularizationContext, RegularizationType}
import com.linkedin.photon.ml.optimization.game._
import com.linkedin.photon.ml.util.DoubleRange

/**
 * Evaluation function implementation for GAME.
 *
 * An evaluation function is the integration point between the hyperparameter tuning module and an estimator, or any
 * system that can unpack a vector of values and produce a real evaluation.
 * @param estimator The estimator for GAME model
 * @param baseConfig The initial configuration supplied by the user
 * @param data Training data
 * @param validationData Validation data
 * @param isOptMax a Boolean indicates that the problem is a maximization (true) or minimization (false).
 */
class GameEstimatorEvaluationFunction(
    estimator: GameEstimator,
    baseConfig: GameOptimizationConfiguration,
    data: DataFrame,
    validationData: DataFrame,
    isOptMax: Boolean)
  extends EvaluationFunction[GameResult] {

  import GameEstimatorEvaluationFunction._

  // CoordinateOptimizationConfigurations sorted in order by coordinate ID name
  private val baseConfigSeq = baseConfig.toSeq.sortBy(_._1)

  // Pull the hyperparameter ranges from the optimization configuration
  protected[estimators] val ranges: Seq[DoubleRange] = baseConfigSeq
    .flatMap {
      case (_, config: GLMOptimizationConfiguration) =>
        val regularizationWeightRange = config
          .regularizationWeightRange
          .getOrElse(DEFAULT_REG_WEIGHT_RANGE)
          .transform(log)

        val elasticNetParamRange = config
          .elasticNetParamRange
          .getOrElse(DEFAULT_REG_ALPHA_RANGE)

        config.regularizationContext.regularizationType match {
          case RegularizationType.ELASTIC_NET =>
            Seq(regularizationWeightRange, elasticNetParamRange)

          case _ =>
            Seq(regularizationWeightRange)
        }

      case _ => Seq()
    }

  // Number of parameters in the base configuration
  val numParams: Int = ranges.length

  /**
   * Performs the evaluation.
   *
   * @param candidate The candidate vector of hyperparameter with values in [0, 1]
   * @return A tuple of the evaluated value and the original output from the inner estimator
   */
  override def apply(candidate: DenseVector[Double]): (Double, GameResult) = {

    val candidateScaled = VectorRescaling.scaleBackward(candidate, ranges)

    val newConfiguration = vectorToConfiguration(candidateScaled)

    val model = estimator.fit(data, Some(validationData), Seq(newConfiguration)).head
    val (_, Some(evaluations), _) = model

    // If this is a maximization problem, flip signs of evaluation values
    val direction = if (isOptMax) -1 else 1

    // Assumes model selection evaluator is in "head" position
    (direction * evaluations.head._2, model)
  }

  /**
   * Vectorize and scale a [[Seq]] of prior observations.
   *
   * @param observations Prior observations in estimator output form
   * @return Prior observations as tuples of (vector representation of the original estimator output, evaluated value)
   */
  override def convertObservations(observations: Seq[GameResult]): Seq[(DenseVector[Double], Double)] = {

    observations.map { observation =>
      val candidate = vectorizeParams(observation)
      val candidateScaled = VectorRescaling.scaleForward(candidate, ranges)

      // If this is a maximization problem, flip signs of evaluation values
      val direction = if (isOptMax) -1 else 1

      val value = direction * getEvaluationValue(observation)

      (candidateScaled, value)
    }
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
      .flatMap { case (_, optConfig) =>
        optConfig match {
          case config: GLMOptimizationConfiguration =>
            config.regularizationContext match {
              // For elastic net, pack weight and alpha
              case RegularizationContext(RegularizationType.ELASTIC_NET, Some(alpha)) =>
                Seq(log(config.regularizationWeight), alpha)

              // Otherwise, just weight
              case _ =>
                Seq(log(config.regularizationWeight))
            }

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
            fixed.regularizationContext.regularizationType match {
              // For elastic net, unpack weight and alpha
              case RegularizationType.ELASTIC_NET => fixed.copy(
                regularizationWeight = exp(paramValues.dequeue()),
                regularizationContext = ElasticNetRegularizationContext(paramValues.dequeue()))

              // Otherwise, just weight
              case _ => fixed.copy(regularizationWeight = exp(paramValues.dequeue()))
            }

          case random: RandomEffectOptimizationConfiguration =>
            random.regularizationContext.regularizationType match {
              // For elastic net, unpack weight and alpha
              case RegularizationType.ELASTIC_NET => random.copy(
                regularizationWeight = exp(paramValues.dequeue()),
                regularizationContext = ElasticNetRegularizationContext(paramValues.dequeue()))

              // Otherwise, just weight
              case _ => random.copy(regularizationWeight = exp(paramValues.dequeue()))
            }

          case other =>
            throw new IllegalArgumentException(s"Unknown coordinate optimization configuration type: ${other.getClass}")
        }

        (coordinateId, newCoordinateConfig)
      }
      .toMap
  }
}

object GameEstimatorEvaluationFunction {
  val DEFAULT_REG_WEIGHT_RANGE = DoubleRange(1e-4, 1e4)
  val DEFAULT_REG_ALPHA_RANGE = DoubleRange(0.0, 1.0)
}
