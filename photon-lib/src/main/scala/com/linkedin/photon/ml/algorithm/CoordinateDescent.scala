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

import scala.collection.mutable

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.storage.StorageLevel
import org.slf4j.Logger

import com.linkedin.photon.ml.Types.{CoordinateId, UniqueSampleId}
import com.linkedin.photon.ml.data.GameDatum
import com.linkedin.photon.ml.data.scoring.CoordinateDataScores
import com.linkedin.photon.ml.evaluation.{EvaluationResults, EvaluationSuite, EvaluatorType}
import com.linkedin.photon.ml.model.{DatumScoringModel, GameModel}
import com.linkedin.photon.ml.optimization.OptimizationTracker
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

  import CoordinateDescent._

  checkInvariants()

  // Coordinates which require model optimization
  private val coordinatesToTrain: Seq[CoordinateId] = updateSequence.filterNot(lockedCoordinates.contains)

  // True number of coordinate descent iterations to perform: if only one coordinate to optimize, reduce iterations to 1
  private val iterations: Int = if (descentIterations > 1 && coordinatesToTrain.size == 1) {
      logger.info(
        s"Given number of coordinate descent iterations is $descentIterations but only one coordinate to train " +
          s"(${coordinatesToTrain.head}): reducing number of iterations to 1")

      1
    } else {
      descentIterations
    }

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

    // At least one coordinate must be trained: cannot have all coordinates be locked
    require(
      updateSequence.exists(!lockedCoordinates.contains(_)),
      "All coordinates in update sequence are locked")
  }

  /**
   * Helper function to make sure that input to coordinate descent is valid.
   *
   * @param coordinates A map of optimization problem coordinates (optimization sub-problems)
   * @param initialModelsOpt An optional map of existing models
   */
  private def checkInput(
      coordinates: Map[CoordinateId, Coordinate[_]],
      initialModelsOpt: Option[Map[CoordinateId, DatumScoringModel]]): Unit = {

    // All coordinates in the update sequence must be passed as input
    updateSequence.foreach { coordinateId =>
      require(
        coordinates.contains(coordinateId),
        s"Coordinate '$coordinateId' in update sequence is not found.")
    }

    // All locked coordinates must have initial models
    require(
      initialModelsOpt match {
        case Some(initialModels) =>
          lockedCoordinates.forall(initialModels.contains)

        case None =>
          lockedCoordinates.isEmpty
      },
      "Missing initial model(s) for locked coordinate(s)")
  }

  /**
   * Run coordinate descent.
   *
   * @param coordinates A map of optimization problem coordinates (optimization sub-problems)
   * @param initialModelsOpt An optional map of existing models
   * @return The best GAME model and its evaluation results, if any (Note: "best" means with the best evaluation metric
   *         score on the validation data; if no validation data is provided, then the "best" model is the GAME model
   *         at the conclusion of coordinate descent).
   */
  def run(
      coordinates: Map[CoordinateId, Coordinate[_]],
      initialModelsOpt: Option[Map[CoordinateId, DatumScoringModel]]): (GameModel, Option[EvaluationResults]) = {

    checkInput(coordinates, initialModelsOpt)

    val initialModels = initialModelsOpt.getOrElse(Map[CoordinateId, DatumScoringModel]())

    // Route coordinate descent based on class parameters
    if (updateSequence.length == 1 && iterations == 1) {
      val coordinateId = updateSequence.head

      descendSingleCoordinate(
        coordinateId,
        coordinates(coordinateId),
        initialModels.get(coordinateId),
        validationDataAndEvaluationSuiteOpt)

    } else if (validationDataAndEvaluationSuiteOpt.isDefined) {
      val (validationData, evaluationSuite) = validationDataAndEvaluationSuiteOpt.get
      val (model, evaluationsResults) = descendWithValidation(
        coordinates,
        updateSequence,
        coordinatesToTrain,
        iterations,
        initialModels,
        validationData,
        evaluationSuite)

      (model, Some(evaluationsResults))

    } else {
      (descend(coordinates, updateSequence, coordinatesToTrain, iterations, initialModels), None)
    }
  }
}

object CoordinateDescent {

  /**
   * Train a new [[DatumScoringModel]] for a [[Coordinate]], using an existing model and/or residual scores if provided.
   *
   * @param coordinateId The ID of the coordinate for which to train a new model
   * @param coordinate The coordinate for which to train a new model
   * @param iteration The current iteration of coordinate descent (for logging purposes)
   * @param initialModelOpt An optional initial model whose coefficients should be used as a starting point for
   *                        optimization
   * @param residualsOpt Optional residual scores to add to the training data offsets
   * @param logger An implicit logger
   * @return The new model trained for the coordinate
   */
  protected[algorithm] def trainCoordinateModel(
      coordinateId: CoordinateId,
      coordinate: Coordinate[_],
      iteration: Int,
      initialModelOpt: Option[DatumScoringModel],
      residualsOpt: Option[CoordinateDataScores])(
      implicit logger: Logger): DatumScoringModel =

    Timed(s"Optimizing coordinate '$coordinateId' for iteration $iteration") {

      logger.debug(s"Updating coordinate of class ${coordinate.getClass}")

      val (model, trackerOpt) = (initialModelOpt, residualsOpt) match {
        case (Some(initialModel), Some(residuals)) =>
          Timed(s"Train new model with residuals using existing model as starting point") {
            coordinate.trainModel(initialModel, residuals)
          }

        case (Some(initialModel), None) =>
          Timed(s"Train new model using existing model as starting point") {
            coordinate.trainModel(initialModel)
          }

        case (None, Some(residuals)) =>
          Timed(s"Train new model with residuals") {
            coordinate.trainModel(residuals)
          }

        case (None, None) =>
          Timed(s"Train new model") {
            coordinate.trainModel()
          }
      }

      logOptimizationSummary(logger, coordinateId, model, trackerOpt)

      model
  }

  /**
   * Train a new [[DatumScoringModel]] for a [[Coordinate]], using an existing model and/or residual scores if provided.
   *
   * @param logger The logger to which to send debug messages
   * @param coordinateId The ID of the coordinate for which to train a new model
   * @param datumScoringModel A newly trained model
   * @param optimizationTrackerOpt An optional tracker of optimization states
   */
  protected[algorithm] def logOptimizationSummary(
      logger: Logger,
      coordinateId: CoordinateId,
      datumScoringModel: DatumScoringModel,
      optimizationTrackerOpt: Option[OptimizationTracker]): Unit = if (logger.isDebugEnabled) {

    logger.debug(s"Summary of coordinate optimization for coordinate $coordinateId:")

    logger.debug(s"Summary of the new model:")
    logger.debug(datumScoringModel.toSummaryString)

    // If optimization tracking is enabled, log the optimization summary
    optimizationTrackerOpt.foreach { optimizationTracker =>
      logger.debug(s"Summary of optimization for the new model:")
      logger.debug(optimizationTracker.toSummaryString)

      optimizationTracker match {
        case rddLike: RDDLike => rddLike.unpersistRDD()
        case _ =>
      }
    }
  }

  /**
   * Train a new [[DatumScoringModel]] for a [[Coordinate]] if it's not locked. Otherwise, return the locked model.
   *
   * @param coordinateId The ID of the coordinate for which to train a new model
   * @param coordinate The coordinate for which to train a new model
   * @param coordinatesToTrain A list of coordinates for which to train new models
   * @param iteration The current iteration of coordinate descent (for logging purposes)
   * @param initialModelOpt An optional initial model whose coefficients should be used as a starting point for
   *                        optimization
   * @param residualsOpt Optional residual scores to add to the training data offsets
   * @param logger An implicit logger
   * @return The locked model if a new model should not be trained for this coordinate, a newly trained model otherwise.
   */
  protected[algorithm] def trainOrFetchCoordinateModel(
      coordinateId: CoordinateId,
      coordinate: Coordinate[_],
      coordinatesToTrain: Seq[CoordinateId],
      iteration: Int,
      initialModelOpt: Option[DatumScoringModel],
      residualsOpt: Option[CoordinateDataScores])(
      implicit logger: Logger): DatumScoringModel =

    if (coordinatesToTrain.contains(coordinateId)) {

      trainCoordinateModel(coordinateId, coordinate, iteration, initialModelOpt, residualsOpt)

    } else {
      logger.info(s"Skipping optimization for locked coordinate '$coordinateId', iteration $iteration")

      initialModelOpt.get
    }

  /**
   * Evaluate a model on validation data using one or more [[com.linkedin.photon.ml.evaluation.Evaluator]] objects.
   *
   * @param modelToEvaluate The model to evaluate
   * @param validationData The data to use for model evaluation
   * @param evaluationSuite The [[com.linkedin.photon.ml.evaluation.Evaluator]] objects to use for model evaluation
   * @param logger An implicit logger
   * @return The evaluation results for the model on the validation data from each
   *         [[com.linkedin.photon.ml.evaluation.Evaluator]]
   */
  protected[algorithm] def evaluateModel(
      modelToEvaluate: DatumScoringModel,
      validationData: RDD[(UniqueSampleId, GameDatum)],
      evaluationSuite: EvaluationSuite)(
      implicit logger: Logger): EvaluationResults = Timed("Validate GAME model") {

    val validatingScores = Timed(s"Compute validation scores") {
      modelToEvaluate.scoreForCoordinateDescent(validationData)
    }

    Timed(s"Compute evaluation metrics") {
      val results = evaluationSuite.evaluate(validatingScores.scoresRdd)

      results
        .evaluations
        .foreach { case (evaluatorType, evaluation) =>
          logger.info(s"${evaluatorType.name}: $evaluation")
        }

      results
    }
  }

  /**
   * Cache a newly trained [[Coordinate]] model to disk.
   *
   * @param model The model to cache
   * @param coordinateId The ID of the coordinate for which the model was trained (for logging purposes)
   * @param iteration The current iteration of coordinate descent (for logging purposes)
   */
  protected[algorithm] def persistModel(model: DatumScoringModel, coordinateId: CoordinateId, iteration: Int): Unit = model match {
    case rddModel: RDDLike =>
      rddModel
        .setName(s"Model for coordinate '$coordinateId', iteration $iteration")
        .persistRDD(StorageLevel.DISK_ONLY)
        .materialize()

    case _ =>
  }

  /**
   * Cache summed residual scores to memory/disk.
   *
   * @param coordinateDataScores The residual scores to cache
   */
  protected[algorithm] def persistSummedScores(coordinateDataScores: CoordinateDataScores): Unit =
    coordinateDataScores.setName(s"Summed scores").persistRDD(StorageLevel.MEMORY_AND_DISK_SER).materialize()

  /**
   * Remove a cached model from cache.
   *
   * @param model The model to remove from cache
   */
  protected[algorithm] def unpersistModel(model: DatumScoringModel): Unit = model match {
    case rddModel: RDDLike =>
      rddModel.unpersistRDD()

    case broadcastModel: BroadcastLike =>
      broadcastModel.unpersistBroadcast()
  }

  /**
   * Perform coordinate descent on the given coordinates to produce a new [[GameModel]]. This function returns the final
   * GAME model after running all iterations of coordinate descent.
   *
   * @note The code of this function and [[descendWithValidation]] are almost identical. However, they have been split
   *       into separate functions to improve readability and to avoid having many conditional objects and code blocks.
   *
   * @param coordinates The [[Coordinate]] objects for which to optimize
   * @param updateSequence The sequence in which to optimize coordinates
   * @param coordinatesToTrain The ordered set of coordinates which are not locked (for which a new model is required)
   * @param iterations The number of coordinate descent iterations, i.e. the number of times to optimize each
   *                   [[Coordinate]] in order of the update sequence
   * @param initialModels A map of existing models; these models are either locked or initial models to use for
   *                      warm-start training
   * @param logger An implicit logger
   * @return A new [[GameModel]]
   */
  private def descend(
      coordinates: Map[CoordinateId, Coordinate[_]],
      updateSequence: Seq[CoordinateId],
      coordinatesToTrain: Seq[CoordinateId],
      iterations: Int,
      initialModels: Map[CoordinateId, DatumScoringModel])(
      implicit logger: Logger): GameModel = {

    var i: Int = 2

    //
    // First coordinate, first iteration
    //

    val firstCoordinateId = updateSequence.head
    val firstCoordinate = coordinates(firstCoordinateId)
    val firstCoordinateModel = trainOrFetchCoordinateModel(
      firstCoordinateId,
      firstCoordinate,
      coordinatesToTrain,
      iteration = 1,
      initialModels.get(firstCoordinateId),
      residualsOpt = None)

    persistModel(firstCoordinateModel, firstCoordinateId, iteration = 1)

    var previousScores = firstCoordinate.score(firstCoordinateModel)
    var summedScores: CoordinateDataScores =
      CoordinateDataScores(SparkSession.builder().getOrCreate().sparkContext.emptyRDD)
    val currentModels: mutable.Map[CoordinateId, DatumScoringModel] =
      mutable.Map(firstCoordinateId -> firstCoordinateModel)
    val currentScores: mutable.Map[CoordinateId, CoordinateDataScores] =
      mutable.Map(firstCoordinateId -> previousScores)

    previousScores.persistRDD(StorageLevel.DISK_ONLY)

    //
    // Subsequent coordinates, first iteration
    //

    updateSequence.tail.foreach { coordinateId =>

      val newSummedScores = previousScores + summedScores
      persistSummedScores(newSummedScores)
      summedScores.unpersistRDD()
      summedScores = newSummedScores

      val coordinate = coordinates(coordinateId)
      val newModel = trainOrFetchCoordinateModel(
        coordinateId,
        coordinate,
        coordinatesToTrain,
        iteration = 1,
        initialModels.get(coordinateId),
        Some(summedScores))

      persistModel(newModel, coordinateId, iteration = 1)

      val scores = coordinate.score(newModel)
      scores.persistRDD(StorageLevel.DISK_ONLY)

      currentModels.put(coordinateId, newModel)
      currentScores.put(coordinateId, scores)
      previousScores = scores
    }

    //
    // Subsequent coordinates, subsequent iterations
    //

    while (i <= iterations) {

      coordinatesToTrain.foreach { coordinateId =>

        val oldScores = currentScores(coordinateId)
        val newSummedScores = summedScores - oldScores + previousScores
        persistSummedScores(newSummedScores)
        summedScores.unpersistRDD()
        oldScores.unpersistRDD()
        summedScores = newSummedScores

        val coordinate = coordinates(coordinateId)
        val oldModelOpt = currentModels.get(coordinateId)
        val newModel = trainCoordinateModel(coordinateId, coordinate, i, oldModelOpt, Some(summedScores))

        persistModel(newModel, coordinateId, i)
        unpersistModel(oldModelOpt.get)

        val scores = coordinate.score(newModel)
        scores.persistRDD(StorageLevel.DISK_ONLY)

        currentModels.put(coordinateId, newModel)
        currentScores.put(coordinateId, scores)
        previousScores = scores
      }

      i += 1
    }

    summedScores.unpersistRDD()
    currentScores.foreach { case (_, scores) =>
      scores.unpersistRDD()
    }

    new GameModel(currentModels.toMap)
  }

  /**
   * Perform coordinate descent on the given coordinates to produce a new [[GameModel]]. This function returns the best
   * model as measured by an evaluation metric.
   *
   * @note The code of this function and [[descend]] are almost identical. However, they have been split into separate
   *       functions to improve readability and to avoid having many conditional objects and code blocks.
   *
   * @param coordinates The [[Coordinate]] objects for which to optimize
   * @param updateSequence The sequence in which to optimize coordinates
   * @param coordinatesToTrain The ordered set of coordinates which are not locked (for which a new model is required)
   * @param iterations The number of coordinate descent iterations, i.e. the number of times to optimize each
   *                   [[Coordinate]] in order of the update sequence
   * @param initialModels A map of existing models; these models are either locked or initial models to use for
   *                      warm-start training
   * @param validationData The validation data used to evaluate trained models
   * @param evaluationSuite The evaluation metrics to compute for trained models
   * @param logger An implicit logger
   * @return A (new [[GameModel]], model [[EvaluationResults]]) tuple
   */
  private def descendWithValidation(
      coordinates: Map[CoordinateId, Coordinate[_]],
      updateSequence: Seq[CoordinateId],
      coordinatesToTrain: Seq[CoordinateId],
      iterations: Int,
      initialModels: Map[CoordinateId, DatumScoringModel],
      validationData: RDD[(UniqueSampleId, GameDatum)],
      evaluationSuite: EvaluationSuite)(
      implicit logger: Logger): (GameModel, EvaluationResults) = {

    val evaluatorType: EvaluatorType = evaluationSuite.primaryEvaluator.evaluatorType

    var i: Int = 2

    //
    // First coordinate, first iteration
    //

    val firstCoordinateId = updateSequence.head
    val firstCoordinate = coordinates(firstCoordinateId)
    val firstCoordinateModel = trainOrFetchCoordinateModel(
      firstCoordinateId,
      firstCoordinate,
      coordinatesToTrain,
      iteration = 1,
      initialModels.get(firstCoordinateId),
      residualsOpt = None)

    persistModel(firstCoordinateModel, firstCoordinateId, iteration = 1)

    var previousScores = firstCoordinate.score(firstCoordinateModel)
    var summedScores: CoordinateDataScores =
      CoordinateDataScores(SparkSession.builder().getOrCreate().sparkContext.emptyRDD)
    val currentModels: mutable.Map[CoordinateId, DatumScoringModel] =
      mutable.Map(firstCoordinateId -> firstCoordinateModel)
    val currentScores: mutable.Map[CoordinateId, CoordinateDataScores] =
      mutable.Map(firstCoordinateId -> previousScores)
    var bestModels: Map[CoordinateId, DatumScoringModel] = currentModels.toMap
    var bestEvaluationResults: EvaluationResults = evaluateModel(
      firstCoordinateModel,
      validationData,
      evaluationSuite)

    previousScores.persistRDD(StorageLevel.DISK_ONLY)

    //
    // Subsequent coordinates, first iteration
    //

    updateSequence.tail.foreach { coordinateId =>

      val newSummedScores = previousScores + summedScores
      persistSummedScores(newSummedScores)
      summedScores.unpersistRDD()
      summedScores = newSummedScores

      val coordinate = coordinates(coordinateId)
      val newModel = trainOrFetchCoordinateModel(
        coordinateId,
        coordinate,
        coordinatesToTrain,
        iteration = 1,
        initialModels.get(coordinateId),
        Some(summedScores))

      persistModel(newModel, coordinateId, iteration = 1)

      val scores = coordinate.score(newModel)
      scores.persistRDD(StorageLevel.DISK_ONLY)

      currentModels.put(coordinateId, newModel)
      currentScores.put(coordinateId, scores)
      previousScores = scores

      val evaluationModel = new GameModel(currentModels.toMap)
      val evaluationResults = evaluateModel(evaluationModel, validationData, evaluationSuite)

      // Log warning if adding a coordinate reduces the overall model performance
      if (evaluatorType.betterThan(bestEvaluationResults.primaryEvaluation, evaluationResults.primaryEvaluation)) {
        logger.info(s"Warning: adding model for coordinate '$coordinateId' reduces overall model performance")
      }

      bestEvaluationResults = evaluationResults
    }

    //
    // Subsequent coordinates, subsequent iterations
    //

    bestModels = currentModels.toMap

    while (i <= iterations) {

      coordinatesToTrain.foreach { coordinateId =>

        val oldScores = currentScores(coordinateId)
        val newSummedScores = summedScores - oldScores + previousScores
        persistSummedScores(newSummedScores)
        summedScores.unpersistRDD()
        oldScores.unpersistRDD()
        summedScores = newSummedScores

        val coordinate = coordinates(coordinateId)
        val oldModelOpt = currentModels.get(coordinateId)
        val newModel = trainCoordinateModel(coordinateId, coordinate, i, oldModelOpt, Some(summedScores))

        persistModel(newModel, coordinateId, i)
        // If the best GAME model doesn't have a model for this coordinate or it does but it's not the old model,
        // unpersist the old model.
        if (bestModels.get(coordinateId).forall(!_.eq(oldModelOpt.get))) {
          unpersistModel(oldModelOpt.get)
        }

        val scores = coordinate.score(newModel)
        scores.persistRDD(StorageLevel.DISK_ONLY)

        currentModels.put(coordinateId, newModel)
        currentScores.put(coordinateId, scores)
        previousScores = scores

        val evaluationModel = new GameModel(currentModels.toMap)
        val evaluationResults = evaluateModel(evaluationModel, validationData, evaluationSuite)
        if (evaluatorType.betterThan(evaluationResults.primaryEvaluation, bestEvaluationResults.primaryEvaluation)) {
          bestEvaluationResults = evaluationResults

          val newBestModels = currentModels.toMap

          // Unpersist each model of the old best GAME model if it's not used in the new best GAME model
          bestModels.foreach { case (bestId, bestModel) =>
            if (!newBestModels(bestId).eq(bestModel)) {
              unpersistModel(bestModel)
            }
          }

          bestModels = newBestModels
        }
      }

      i += 1
    }

    summedScores.unpersistRDD()
    currentScores.foreach { case (_, scores) =>
      scores.unpersistRDD()
    }
    currentModels.foreach { case (coordinateId, model) =>
      // If the best GAME model doesn't have a model for this coordinate or it does but they don't match, unpersist it
      if (bestModels.get(coordinateId).forall(!_.eq(model))) {
        unpersistModel(model)
      }
    }

    (new GameModel(bestModels), bestEvaluationResults)
  }

  /**
   * Train a new [[GameModel]] made up of a single coordinate.
   *
   * @param coordinateId The ID of the single coordinate for which to train a new model
   * @param coordinate The [[Coordinate]] for which to train a new model
   * @param initialModelOpt An optional existing model to use for warm-start training
   * @param validationDataAndEvaluationSuiteOpt An optional (validation data, set of evaluation metrics to compute)
   *                                            tuple
   * @param logger An implicit logger
   * @return A (new [[GameModel]], optional model [[EvaluationResults]]) tuple
   */
  private def descendSingleCoordinate(
      coordinateId: CoordinateId,
      coordinate: Coordinate[_],
      initialModelOpt: Option[DatumScoringModel],
      validationDataAndEvaluationSuiteOpt: Option[(RDD[(UniqueSampleId, GameDatum)], EvaluationSuite)])(
      implicit logger: Logger): (GameModel, Option[EvaluationResults]) = {

    val newModel = trainCoordinateModel(coordinateId, coordinate, iteration = 1, initialModelOpt, residualsOpt = None)

    persistModel(newModel, coordinateId, iteration = 1)

    val evaluationResultsOpt = validationDataAndEvaluationSuiteOpt.map { case (validationData, evaluationSuite) =>
      evaluateModel(newModel, validationData, evaluationSuite)
    }

    (new GameModel(Map(coordinateId -> newModel)), evaluationResultsOpt)
  }
}
