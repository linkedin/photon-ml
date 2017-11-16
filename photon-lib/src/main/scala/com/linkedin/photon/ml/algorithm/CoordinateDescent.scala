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
package com.linkedin.photon.ml.algorithm

import org.apache.spark.rdd.RDD
import org.slf4j.Logger

import com.linkedin.photon.ml.Types.{CoordinateId, UniqueSampleId}
import com.linkedin.photon.ml.constants.{MathConst, StorageLevel}
import com.linkedin.photon.ml.data.GameDatum
import com.linkedin.photon.ml.evaluation.Evaluator
import com.linkedin.photon.ml.evaluation.Evaluator.EvaluationResults
import com.linkedin.photon.ml.model.GameModel
import com.linkedin.photon.ml.spark.{BroadcastLike, RDDLike}
import com.linkedin.photon.ml.util.Timed

/**
 * Coordinate descent implementation.
 *
 * @param coordinates The individual optimization problem coordinates
 * @param trainingLossFunctionEvaluator Training loss function evaluator
 * @param validationDataAndEvaluatorsOption Optional validation data and evaluator
 * @param logger A logger instance
 */
class CoordinateDescent(
    coordinates: Seq[(CoordinateId, Coordinate[_])],
    trainingLossFunctionEvaluator: Evaluator,
    validationDataAndEvaluatorsOption: Option[(RDD[(UniqueSampleId, GameDatum)], Seq[Evaluator])],
    implicit private val logger: Logger) {

  import CoordinateDescent._

  /**
   * Run coordinate descent.
   *
   * @param descentIterations Number of coordinate descent iterations (updates to each coordinate in order)
   * @param seed Random seed (default: MathConst.RANDOM_SEED)
   * @return A trained GAME model
   */
  def run(descentIterations: Int, seed: Long = MathConst.RANDOM_SEED): (GameModel, Option[EvaluationResults]) = {

    val initializedModelContainer = coordinates
      .map { case (coordinateId, coordinate) =>
        val initializedModel = coordinate.initializeModel(seed)

        initializedModel match {
          case rddLike: RDDLike =>
            rddLike
              .setName(s"Initialized model with coordinate id $coordinateId")
              .persistRDD(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)

          case _ =>
        }

        if (logger.isDebugEnabled) {
          logger.debug(
            s"""Summary of model (${initializedModel.getClass}) initialized for coordinate with ID $coordinateId:
              |${initializedModel.toSummaryString}
              |""".stripMargin)
        }

        (coordinateId, initializedModel)
      }
      .toMap

    val initialGameModel = new GameModel(initializedModelContainer)
    run(descentIterations, initialGameModel)
  }

  /**
   * This function optimizes the model w.r.t. the objective function. Optionally, it also evaluates the model on the
   * validation data set using one or more validation functions. In that case, the output is the model which yielded the
   * best evaluation on the validation data set w.r.t. the primary evaluation function. Otherwise, it's simply the
   * trained model.
   *
   * @param descentIterations Number of coordinate descent iterations (updates to each coordinate in order)
   * @param gameModel The initial GAME model
   * @return The best GAME model (see above for exact meaning of "best")
   */
  def run(descentIterations: Int, gameModel: GameModel): (GameModel, Option[EvaluationResults]) = {

    //
    // Input verification
    //

    // Verify valid number of descent iterations
    require(descentIterations > 0, s"Number of coordinate descent iterations must be greater than 0: $descentIterations")
    // Verify that the model being optimized has entries for each coordinate
    coordinates.foreach { case (coordinateId, _) =>
      require(
        gameModel.getModel(coordinateId).isDefined,
        s"Model with coordinateId $coordinateId is expected but not found from the initial GAME model")
    }

    //
    // Optimization setup
    //

    var updatedGameModel = gameModel

    // Initialize the training scores
    var updatedScoresContainer = coordinates
      .map { case (coordinateId, coordinate) =>
        val updatedScores = coordinate.score(updatedGameModel.getModel(coordinateId).get)
        updatedScores
          .setName(s"Initialized training scores with coordinateId $coordinateId")
          .persistRDD(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)
          .materialize()

        (coordinateId, updatedScores)
      }
      .toMap
    var fullTrainingScore = updatedScoresContainer.values.reduce(_ + _)
    fullTrainingScore.persistRDD(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL).materialize()

    // Initialize the regularization term value
    var regularizationTermValueContainer = coordinates
      .map { case (coordinateId, coordinate) =>
        (coordinateId, coordinate.computeRegularizationTermValue(updatedGameModel.getModel(coordinateId).get))
      }
      .toMap
    var fullRegularizationTermValue = regularizationTermValueContainer.values.sum

    // Initialize the validation scores
    var validationScoresContainerOption = validationDataAndEvaluatorsOption.map { case (validatingData, _) =>
      coordinates
        .map { case (coordinateId, _) =>
          val updatedModel = updatedGameModel.getModel(coordinateId).get
          val validatingScores = updatedModel.scoreForCoordinateDescent(validatingData)
          validatingScores
            .setName(s"Initialized validating scores with coordinateId $coordinateId")
            .persistRDD(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)
            .materialize()

          (coordinateId, validatingScores)
        }
        .toMap
    }
    var fullValidationScoreOption = validationScoresContainerOption.map(_.values.reduce(_ + _))
    fullValidationScoreOption.map(_.persistRDD(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL).materialize())

    /*
     * This will track the "best" model according to the first evaluation function chosen by the user.
     * If the user did not specify any evaluation function, this var will be None.
     *
     * NOTE: These two variables are updated by comparing *FULL* models, i.e. models after a full update sequence.
     * If we allowed the "best" model to be selected inside the loop over coordinates, the "best" model could be one
     * that doesn't contain some fixed/random effects.
     */
    var bestModel: Option[GameModel] = None
    var bestEvals: Option[EvaluationResults] = None

    //
    // Optimization
    //

    for (iteration <- 0 until descentIterations) {
      Timed(s"Coordinate descent iteration $iteration") {

        val oldGameModel = updatedGameModel
        var unpersistOldGameModel = true

        coordinates
          .map { case (coordinateId, coordinate) =>
            Timed(s"Update coordinate $coordinateId") {

              //
              // Update the coordinate model
              //

              logger.debug(s"Update coordinate with ID $coordinateId (${coordinate.getClass})")

              val oldModel = updatedGameModel.getModel(coordinateId).get
              val (updatedModel, optimizationTrackerOption) = Timed(s"Train coordinate $coordinateId") {
                if (updatedScoresContainer.keys.size > 1) {
                  // If there are multiple coordinates, update using a partial score from the other coordinates.
                  val partialScore = fullTrainingScore - updatedScoresContainer(coordinateId)
                  coordinate.updateModel(oldModel, partialScore)
                } else {
                  // Otherwise, just update the only coordinate
                  coordinate.updateModel(oldModel)
                }
              }

              updatedModel match {
                case rddLike: RDDLike =>
                  rddLike
                    .setName(s"Updated model with coordinateId $coordinateId at iteration $iteration")
                    .persistRDD(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)
                    .materialize()

                case _ =>
              }
              updatedGameModel = updatedGameModel.updateModel(coordinateId, updatedModel)

              //
              // Log coordinate update details
              //

              // Log the objective value for the current GAME model
              val updatedScores = coordinate.score(updatedModel)
              updatedScores
                .setName(s"Updated training scores with key $coordinateId at iteration $iteration")
                .persistRDD(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)
                .materialize()
              val newTrainingScore = fullTrainingScore - updatedScoresContainer(coordinateId) + updatedScores

              updatedScoresContainer(coordinateId).unpersistRDD()
              updatedScoresContainer = updatedScoresContainer.updated(coordinateId, updatedScores)
              newTrainingScore.persistRDD(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL).materialize()
              fullTrainingScore.unpersistRDD()
              fullTrainingScore = newTrainingScore

              // Log a summary of the updated model and its objective value. Do a check for debug first, to not waste
              // time computing the summary.
              if (logger.isDebugEnabled) {
                logger.debug(s"Summary of the learned model:\n${updatedModel.toSummaryString}")

                // If optimization tracking is enabled, log the optimization summary
                optimizationTrackerOption.foreach { optimizationTracker =>
                  logger.debug(s"OptimizationTracker:\n${optimizationTracker.toSummaryString}")

                  optimizationTracker match {
                    case rddLike: RDDLike => rddLike.unpersistRDD()
                    case _ =>
                  }
                }

                val updatedRegularizationTermValue = coordinate.computeRegularizationTermValue(updatedModel)
                fullRegularizationTermValue = (fullRegularizationTermValue
                  - regularizationTermValueContainer(coordinateId)
                  + updatedRegularizationTermValue)
                regularizationTermValueContainer =
                  regularizationTermValueContainer.updated(coordinateId, updatedRegularizationTermValue)

                logger.debug(s"Objective value after updating coordinate $coordinateId, iteration $iteration:")
                logger.debug(
                  formatObjectiveValue(
                    trainingLossFunctionEvaluator.evaluate(fullTrainingScore.scores),
                    fullRegularizationTermValue))
              }

              //
              // Validate the updated GAME model
              //

              // Update the validation score and evaluate the updated model on the validating data
              validationDataAndEvaluatorsOption.map { case (validatingData, evaluators) =>
                Timed("Validate GAME model") {
                  val validatingScoresContainer = validationScoresContainerOption.get
                  val validatingScores = updatedModel.scoreForCoordinateDescent(validatingData)
                  validatingScores
                    .setName(s"Updated validating scores with coordinateId $coordinateId")
                    .persistRDD(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)
                    .materialize()
                  val fullValidationScore = (fullValidationScoreOption.get
                    - validatingScoresContainer(coordinateId)
                    + validatingScores)
                  fullValidationScore.persistRDD(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL).materialize()
                  fullValidationScoreOption.get.unpersistRDD()
                  fullValidationScoreOption = Some(fullValidationScore)
                  validatingScoresContainer(coordinateId).unpersistRDD()
                  val updatedValidatingScoresContainer = validatingScoresContainer.updated(coordinateId, validatingScores)
                  validationScoresContainerOption = Some(updatedValidatingScoresContainer)

                  evaluators.map { evaluator =>
                    val evaluation = Timed(s"Evaluate with ${evaluator.getEvaluatorName}") {
                      evaluator.evaluate(fullValidationScore.scores)
                    }

                    logger.info(s"Evaluation metric computed with ${evaluator.getEvaluatorName} after updating " +
                      s"coordinateId $coordinateId at iteration $iteration is $evaluation")

                    (evaluator, evaluation)
                  }
                }
              }
            }
          } // End of coordinates update
          .last
          .foreach { evaluations =>
            if (evaluations.nonEmpty) {
              // The first evaluator is used for model selection
              val (evaluator, evaluation) = evaluations.head

              if (bestEvals.forall(e => evaluator.betterThan(evaluation, e.head._2))) {

                // Unpersist the previous best models
                bestModel.foreach { gameModel =>
                  gameModel.toMap.foreach { case (_, model) =>
                    // We need to split out the following 2 match expressions: [[FactoredRandomEffectModel]] matches both
                    model match {
                      case broadcastLike: BroadcastLike => broadcastLike.unpersistBroadcast()
                      case _ =>
                    }
                    model match {
                      case rddLike: RDDLike => rddLike.unpersistRDD()
                      case _ =>
                    }
                  }
                }

                bestEvals = Some(evaluations)
                bestModel = Some(updatedGameModel)

                logger.debug(s"Found better GAME model, with evaluation: $evaluation")

              } else {

                // We always want to unpersist the previous GAME model UNLESS the current best GAME model is the old
                // model
                unpersistOldGameModel = bestModel.forall(_ != oldGameModel)

                logger.debug(
                  s"Previous iterations GAME model is better. Ignoring GAME model with evaluation: $evaluation")
              }

            } else {

              logger.debug("No evaulator specified to select best model")
            }
          }

        // Unpersist the previous GAME model
        if (unpersistOldGameModel) {
          oldGameModel.toMap.foreach { case (_, model) =>
            // We need to split out the following 2 match expressions: [[FactoredRandomEffectModel]] matches both
            model match {
              case broadcastLike: BroadcastLike => broadcastLike.unpersistBroadcast()
              case _ =>
            }
            model match {
              case rddLike: RDDLike => rddLike.unpersistRDD()
              case _ =>
            }
          }
        }
      }
    } // end optimization

    updatedScoresContainer.mapValues(_.unpersistRDD())
    fullTrainingScore.unpersistRDD()
    validationScoresContainerOption.map(_.mapValues(_.unpersistRDD()))
    fullValidationScoreOption.map(_.unpersistRDD())

    (bestModel.getOrElse(updatedGameModel), bestEvals)
  }
}

object CoordinateDescent {

  /**
   * Helper function to format objective function value for logging purposes.
   *
   * @param lossFunctionValue The value of the loss function
   * @param regularizationTermValue The value of the regularization term
   * @return The two input values and the total objective function value, formatted for output to logs
   */
  private def formatObjectiveValue(lossFunctionValue: Double, regularizationTermValue: Double): String = {
    val objectiveFunctionValue = lossFunctionValue + regularizationTermValue

    s"lossFunctionValue: $lossFunctionValue, regularizationTermValue: $regularizationTermValue, " +
      s"objectiveFunctionValue: $objectiveFunctionValue"
  }

  /**
   * Helper function to create a new [[CoordinateDescent]] instance.
   *
   * @param coordinates The individual optimization problem coordinates
   * @param trainingLossFunctionEvaluator Training loss function evaluator
   * @param validationDataAndEvaluatorsOption Optional validation data and evaluator
   * @param logger A logger instance
   * @return A new [[CoordinateDescent]] instance
   */
  def apply(
      coordinates: Seq[(CoordinateId, Coordinate[_])],
      trainingLossFunctionEvaluator: Evaluator,
      validationDataAndEvaluatorsOption: Option[(RDD[(UniqueSampleId, GameDatum)], Seq[Evaluator])],
      logger: Logger) =
    new CoordinateDescent(coordinates, trainingLossFunctionEvaluator, validationDataAndEvaluatorsOption, logger)
}
