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
import org.apache.spark.storage.StorageLevel
import org.slf4j.Logger

import com.linkedin.photon.ml.Types.{CoordinateId, UniqueSampleId}
import com.linkedin.photon.ml.data.GameDatum
import com.linkedin.photon.ml.evaluation.{EvaluationResults, EvaluationSuite}
import com.linkedin.photon.ml.model.GameModel
import com.linkedin.photon.ml.spark.{BroadcastLike, RDDLike}
import com.linkedin.photon.ml.util.Timed

/**
 * Coordinate descent implementation.
 *
 * @param updateSequence The order in which to update coordinates
 * @param descentIterations Number of coordinate descent iterations (updates to each coordinate in order)
 * @param validationDataAndEvaluationSuiteOpt Optional validation data and [[EvaluationSuite]] of validation metric
 *                                         [[com.linkedin.photon.ml.evaluation.Evaluator]] objects
 * @param lockedCoordinates Set of locked coordinates within the initial model for performing partial retraining
 * @param logger A logger instance
 */
class CoordinateDescent(
    updateSequence: Seq[CoordinateId],
    descentIterations: Int,
    validationDataAndEvaluationSuiteOpt: Option[(RDD[(UniqueSampleId, GameDatum)], EvaluationSuite)],
    lockedCoordinates: Set[CoordinateId],
    implicit private val logger: Logger) {

  private val coordinatesToTrain: Seq[CoordinateId] = updateSequence.filterNot(lockedCoordinates.contains)

  checkInvariants()

  /**
   * Invariants that hold for every instance of [[CoordinateDescent]].
   */
  private def checkInvariants(): Unit = {

    // Must have strictly positive number of coordinate descent iterations
    require(
      descentIterations > 0,
      s"Number of coordinate descent iterations must be greater than 0: $descentIterations")

    // Coordinates in the update sequence must not repeat
    require(
      updateSequence.toSet.size == updateSequence.size,
      "One or more coordinates in the update sequence is repeated")
    // All locked coordinates must be present in the update sequence
    require(
      lockedCoordinates.forall(updateSequence.contains),
      "One or more locked coordinates is missing from the update sequence")
  }

  /**
   * Helper function to make sure that input to coordinate descent is valid.
   *
   * @param coordinates A map of optimization problem coordinates (optimization sub-problems)
   */
  private def checkInput(coordinates: Map[CoordinateId, Coordinate[_]]): Unit = {

    // Coordinates in the update sequence must have a corresponding coordinate
    updateSequence.foreach { coordinateId =>
      require(
        coordinates.contains(coordinateId),
        s"Coordinate '$coordinateId' in update sequence is not found in either the coordinates map or the locked " +
          "coordinates.")
    }
  }

  /**
   * Run coordinate descent.
   *
   * @param coordinates A map of optimization problem coordinates (optimization sub-problems)
   * @param gameModel The initial GAME model to use as a starting point
   * @return The best GAME model (see [[CoordinateDescent.descend]] for exact meaning of "best") and its evaluation
   *         results (if any)
   */
  def run(
      coordinates: Map[CoordinateId, Coordinate[_]],
      gameModel: GameModel): (GameModel, Option[EvaluationResults]) = {

    checkInput(coordinates)

    // Verify that the model being optimized has entries for each coordinate
    updateSequence.foreach { coordinateId =>
      require(
        gameModel.getModel(coordinateId).isDefined,
        s"Model with coordinateId $coordinateId is expected but not found from the initial GAME model.")
    }

    descend(coordinates, gameModel)
  }

  /**
   * This function optimizes the model w.r.t. the objective function. Optionally, it also evaluates the model on the
   * validation dataset using one or more validation functions. In that case, the output is the model which yielded the
   * best evaluation on the validation dataset w.r.t. the primary evaluation function. Otherwise, it's simply the
   * trained model.
   *
   * @param coordinates A map of optimization problem coordinates (optimization sub-problems)
   * @param gameModel The initial GAME model to use as a starting point
   * @return The best GAME model (see above for exact meaning of "best") and its evaluation results (if any)
   */
  private def descend(
      coordinates: Map[CoordinateId, Coordinate[_]],
      gameModel: GameModel): (GameModel, Option[EvaluationResults]) = {

    //
    // Optimization setup
    //

    var updatedGameModel = gameModel

    // Initialize the training scores
    var updatedScoresContainer = updateSequence
      .map { coordinateId =>
        val updatedScores = coordinates(coordinateId).score(updatedGameModel.getModel(coordinateId).get)
        updatedScores
          .setName(s"Initialized training scores with coordinateId $coordinateId")
          .persistRDD(StorageLevel.DISK_ONLY)
          .materialize()

        (coordinateId, updatedScores)
      }
      .toMap
    var fullTrainingScore = updatedScoresContainer.values.reduce(_ + _)
    fullTrainingScore.persistRDD(StorageLevel.DISK_ONLY).materialize()

    // Initialize the validation scores
    var validationScoresContainerOption = validationDataAndEvaluationSuiteOpt.map { case (validationData, _) =>
      updateSequence
        .map { coordinateId =>
          val updatedModel = updatedGameModel.getModel(coordinateId).get
          val validationScores = updatedModel.scoreForCoordinateDescent(validationData)
          validationScores
            .setName(s"Initialized validating scores with coordinateId $coordinateId")
            .persistRDD(StorageLevel.DISK_ONLY)
            .materialize()

          (coordinateId, validationScores)
        }
        .toMap
    }
    var fullValidationScoreOption = validationScoresContainerOption.map(_.values.reduce(_ + _))
    fullValidationScoreOption.map(_.persistRDD(StorageLevel.DISK_ONLY).materialize())

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

        coordinatesToTrain
          .map { coordinateId =>
            Timed(s"Update coordinate $coordinateId") {

              val coordinate = coordinates(coordinateId)

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
                    .persistRDD(StorageLevel.DISK_ONLY)
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
                .persistRDD(StorageLevel.DISK_ONLY)
                .materialize()
              val newTrainingScore = fullTrainingScore - updatedScoresContainer(coordinateId) + updatedScores

              updatedScoresContainer(coordinateId).unpersistRDD()
              updatedScoresContainer = updatedScoresContainer.updated(coordinateId, updatedScores)
              newTrainingScore.persistRDD(StorageLevel.DISK_ONLY).materialize()
              fullTrainingScore.unpersistRDD()
              fullTrainingScore = newTrainingScore

              // Log a summary of the updated model and its objective value. Do a check for debug first, to not waste
              // time computing the summary.
              if (logger.isDebugEnabled) {
                logger.debug(s"Summary of the learned model:\n${updatedModel.toSummaryString}")

                // If optimization tracking is enabled, log the optimization summary
                optimizationTrackerOption.foreach { optimizationTracker =>
                  logger.debug(s"Summary of optimization:\n${optimizationTracker.toSummaryString}")
                }
              }

              //
              // Validate the updated GAME model
              //

              // Update the validation score and evaluate the updated model on the validating data
              validationDataAndEvaluationSuiteOpt.map { case (validationData, evaluationSuite) =>
                Timed("Validate GAME model") {
                  val validatingScoresContainer = validationScoresContainerOption.get
                  val validatingScores = updatedModel.scoreForCoordinateDescent(validationData)
                  validatingScores
                    .setName(s"Updated validating scores with coordinateId $coordinateId")
                    .persistRDD(StorageLevel.DISK_ONLY)
                    .materialize()
                  val fullValidationScore = (fullValidationScoreOption.get
                    - validatingScoresContainer(coordinateId)
                    + validatingScores)
                  fullValidationScore.persistRDD(StorageLevel.DISK_ONLY).materialize()
                  fullValidationScoreOption.get.unpersistRDD()
                  fullValidationScoreOption = Some(fullValidationScore)
                  validatingScoresContainer(coordinateId).unpersistRDD()
                  val updatedValidatingScoresContainer = validatingScoresContainer.updated(coordinateId, validatingScores)
                  validationScoresContainerOption = Some(updatedValidatingScoresContainer)

                  Timed(s"Compute validation metrics") {
                    val results = evaluationSuite.evaluate(fullValidationScore.scores)

                    results
                      .evaluations
                      .foreach { case (evaluator, evaluation) =>
                        logger.info(
                          s"Evaluation metric '${evaluator.name}' after updating coordinate '$coordinateId' during " +
                            s"iteration $iteration: $evaluation")
                      }

                    results
                  }
                }
              }
            }
          } // End of coordinates update
          .last
          .foreach { evaluations =>
            val evaluator = evaluations.primaryEvaluator
            val evaluation = evaluations.primaryEvaluation

            if (bestEvals.forall(e => evaluator.betterThan(evaluation, e.primaryEvaluation))) {

              // Unpersist the previous best models
              bestModel.foreach { currBestModel =>
                // Make sure we don't unpersist a model that was passed in from elsewhere
                if (currBestModel.eq(gameModel)) {
                  coordinatesToTrain.foreach { coordinateId =>
                    gameModel.getModel(coordinateId) match {
                      case Some(broadcastLike: BroadcastLike) => broadcastLike.unpersistBroadcast()
                      case Some(rddLike: RDDLike) => rddLike.unpersistRDD()
                      case _ =>
                    }
                  }
                }
              }

              bestEvals = Some(evaluations)
              bestModel = Some(updatedGameModel)

              logger.debug(s"Found better GAME model, with evaluation: $evaluation")

            } else {
              // Mark the previous GAME model to be unpersisted, unless the previous GAME model is the best model
              unpersistOldGameModel = bestModel.forall(_ != oldGameModel)

              logger.debug(
                s"New GAME model does not improve evaluation; ignoring GAME model with evaluation $evaluation")
            }
          }

        // Unpersist the previous GAME model if it has been marked AND if it is not an initial model input
        if (unpersistOldGameModel && !oldGameModel.eq(gameModel)) {
          coordinatesToTrain.foreach { coordinateId =>
            oldGameModel.getModel(coordinateId) match {
              case Some(broadcastLike: BroadcastLike) => broadcastLike.unpersistBroadcast()
              case Some(rddLike: RDDLike) => rddLike.unpersistRDD()
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
