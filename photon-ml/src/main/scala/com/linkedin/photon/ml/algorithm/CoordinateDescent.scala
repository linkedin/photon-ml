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
package com.linkedin.photon.ml.algorithm

import com.linkedin.photon.ml.constants.{MathConst, StorageLevel}
import com.linkedin.photon.ml.data.{DataSet, GameDatum}
import com.linkedin.photon.ml.evaluation.Evaluator
import com.linkedin.photon.ml.model.GAMEModel
import com.linkedin.photon.ml.util.{ObjectiveFunctionValue, PhotonLogger, Timer}
import com.linkedin.photon.ml.{BroadcastLike, RDDLike}
import org.apache.spark.rdd.RDD

/**
  * Coordinate descent implementation
  *
  * @param coordinates The individual optimization problem coordinates. The coordinates is a [[Seq]] consists of
  *                    (coordinateName, [[Coordinate]] object) pairs.
  * @param trainingLossFunctionEvaluator Training loss function evaluator
  * @param validatingDataAndEvaluatorOption Optional validation data evaluator. The validating data is a [[RDD]]
  *                                         consists of (global Id, [[GameDatum]] object pairs), there the global Id
  *                                         is a unique identifier for each [[GameDatum]] object.
  * @param logger A logger instance
  */
class CoordinateDescent(
    coordinates: Seq[(String, Coordinate[_ <: DataSet[_], _ <: Coordinate[_, _]])],
    trainingLossFunctionEvaluator: Evaluator,
    validatingDataAndEvaluatorOption: Option[(RDD[(Long, GameDatum)], Evaluator)],
    logger: PhotonLogger) {

  /**
    * Run coordinate descent
    *
    * @param numIterations Number of iterations
    * @param seed Random seed (default: MathConst.RANDOM_SEED)
    * @return A trained GAME model
    */
  def run(numIterations: Int, seed: Long = MathConst.RANDOM_SEED): GAMEModel = {
    val initializedModelContainer = coordinates.map { case (coordinateId, coordinate) =>
      val initializedModel = coordinate.initializeModel(seed)
      initializedModel match {
        case rddLike: RDDLike =>
          rddLike
            .setName(s"Initialized model with coordinate id $coordinateId")
            .persistRDD(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)
        case _ =>
      }
      logger.debug(s"Summary of model (${initializedModel.getClass}}) initialized for coordinate with " +
          s"ID $coordinateId:\n${initializedModel.toSummaryString}\n")
      (coordinateId, initializedModel)
    }.toMap

    val initialGAMEModel = new GAMEModel(initializedModelContainer)
    run(numIterations, initialGAMEModel)
  }

  /**
    * Run coordinate descent
    *
    * @param numIterations Number of iterations
    * @param gameModel The initial GAME model
    * @return Trained GAME model
    */
  def run(numIterations: Int, gameModel: GAMEModel): GAMEModel = {

    coordinates.foreach { case (coordinateId, _) =>
      require(gameModel.getModel(coordinateId).isDefined,
        s"Model with coordinateId $coordinateId is expected but not found from the initial GAME model!")
    }
    var updatedGAMEModel = gameModel

    // Initialize the training scores
    var updatedScoresContainer = coordinates.map { case (coordinateId, coordinate) =>
      val updatedScores = coordinate.score(updatedGAMEModel.getModel(coordinateId).get)
      updatedScores.setName(s"Initialized training scores with coordinateId $coordinateId")
          .persistRDD(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)
      (coordinateId, updatedScores)
    }.toMap

    // Initialize the validating scores
    var validatingScoresContainerOption = validatingDataAndEvaluatorOption.map { case (validatingData, _) =>
      val validatingScoresContainer = coordinates.map { case (coordinateId, _) =>
        val updatedModel = updatedGAMEModel.getModel(coordinateId).get
        val validatingScores = updatedModel.score(validatingData)
        validatingScores.setName(s"Initialized validating scores with coordinateId $coordinateId")
            .persistRDD(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)
        (coordinateId, validatingScores)
      }.toMap
      validatingScoresContainer
    }

    // Initialize the regularization term value
    var regularizationTermValueContainer = coordinates
      .map { case (coordinateId, coordinate) =>
        val updatedModel = updatedGAMEModel.getModel(coordinateId).get
        (coordinateId, coordinate.computeRegularizationTermValue(updatedModel))
      }
      .toMap

    for (iteration <- 0 until numIterations) {
      val iterationTimer = Timer.start()
      logger.debug(s"Iteration $iteration of coordinate descent starts...\n")
      coordinates.foreach { case (coordinateId, coordinate) =>

        val coordinateTimer = Timer.start()
        logger.debug(s"Start to update coordinate with ID $coordinateId (${coordinate.getClass})")

        // Update the model
        val modelUpdatingTimer = Timer.start()

        val oldModel = updatedGAMEModel.getModel(coordinateId).get
        val (updatedModel, optimizationTracker) = if (updatedScoresContainer.keys.size > 1) {
          // If there are other coordinates, collect their scores into a partial score and optimize
          val partialScore = updatedScoresContainer.filterKeys(_ != coordinateId).values.reduce(_ + _)
          coordinate.updateModel(oldModel, partialScore)

        } else {
          // Otherwise, just optimize
          coordinate.updateModel(oldModel)
        }

        updatedModel match {
          case rddLike: RDDLike =>
            rddLike.setName(s"Updated model with coordinateId $coordinateId at iteration $iteration")
                .persistRDD(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL).materialize()
          case _ =>
        }
        oldModel match {
          case broadcastLike: BroadcastLike => broadcastLike.unpersistBroadcast()
          case _ =>
        }
        oldModel match {
          case rddLike: RDDLike => rddLike.unpersistRDD()
          case _ =>
        }
        updatedGAMEModel = updatedGAMEModel.updateModel(coordinateId, updatedModel)

        // Summarize the current progress
        logger.info(s"Finished training the model in coordinate $coordinateId, " +
            s"time elapsed: ${modelUpdatingTimer.stop().durationSeconds} (s).")
        logger.debug(s"OptimizationTracker:\n${optimizationTracker.toSummaryString}")
        logger.debug(s"Summary of the learned model:\n${updatedModel.toSummaryString}")
        optimizationTracker match {
          case rddLike: RDDLike => rddLike.unpersistRDD()
          case _ =>
        }

        // Update the training score
        val updatedScores = coordinate.score(updatedModel)
        updatedScores.setName(s"Updated training scores with key $coordinateId at iteration $iteration")
            .persistRDD(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)
            .materialize()
        updatedScoresContainer(coordinateId).unpersistRDD()
        updatedScoresContainer = updatedScoresContainer.updated(coordinateId, updatedScores)

        // Update the regularization term value
        val updatedRegularizationTermValue = coordinate.computeRegularizationTermValue(updatedModel)
        regularizationTermValueContainer =
            regularizationTermValueContainer.updated(coordinateId, updatedRegularizationTermValue)
        // Compute the training objective function value
        val fullScore = updatedScoresContainer.values.reduce(_ + _)
        val lossFunctionValue = trainingLossFunctionEvaluator.evaluate(fullScore.scores)
        val regularizationTermValue = regularizationTermValueContainer.values.sum
        val objectiveFunctionValue = ObjectiveFunctionValue(lossFunctionValue, regularizationTermValue)
        logger.info(s"Training objective function value after updating coordinate with id $coordinateId at " +
            s"iteration $iteration is:\n$objectiveFunctionValue")

        // Update the validating score and evaluate the updated model on the validating data
        validatingScoresContainerOption = validatingDataAndEvaluatorOption.map { case (validatingData, evaluator) =>
          val validationTimer = Timer.start()
          var validatingScoresContainer = validatingScoresContainerOption.get
          val validatingScores = updatedModel
            .score(validatingData)
            .setName(s"Updated validating scores with coordinateId $coordinateId")
            .persistRDD(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)
            .materialize()
          validatingScoresContainer(coordinateId).unpersistRDD()
          validatingScoresContainer = validatingScoresContainer.updated(coordinateId, validatingScores)
          val fullScore = validatingScoresContainer.values.reduce(_ + _)
          val evaluationMetric = evaluator.evaluate(fullScore.scores)
          logger.debug(s"Finished validating the model, time elapsed: ${validationTimer.stop().durationSeconds} (s).")
          logger.info(s"Evaluation metric after updating coordinateId $coordinateId at iteration $iteration is " +
              s"$evaluationMetric")
          validatingScoresContainer
        }

        logger.info(
          s"Updating coordinate $coordinateId finished, time elapsed: ${coordinateTimer.stop().durationSeconds} (s)\n")
      }

      logger.info(
        s"Iteration $iteration of coordinate descent finished, time elapsed: " +
        s"${iterationTimer.stop().durationSeconds} (s)\n\n")
    }

    updatedGAMEModel
  }
}
