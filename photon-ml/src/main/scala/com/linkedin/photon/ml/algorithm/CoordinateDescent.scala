package com.linkedin.photon.ml.algorithm

import scala.collection.Map
import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.{BroadcastLike, RDDLike}
import com.linkedin.photon.ml.constants.{StorageLevel, MathConst}
import com.linkedin.photon.ml.data.{DataSet, GameData}
import com.linkedin.photon.ml.evaluation.Evaluator
import com.linkedin.photon.ml.model.Model
import com.linkedin.photon.ml.util.{ObjectiveFunctionValue, PhotonLogger}

/**
 * Coordinate descent implementation
 *
 * @param coordinates the individual optimization problem coordinates. The coordinates is a [[Seq]] consists of
 *                    (coordinateName, [[Coordinate]] object) pairs.
 * @param trainingLossFunctionEvaluator training loss function evaluator
 * @param validatingDataAndEvaluatorOption optional validation data evaluator. The validating data is a [[RDD]] consists
 *                                         of (global Id, [[GameData]] object pairs), there the global Id is a unique
 *                                         identifier for each [[GameData]] object.
 * @param logger logger instance
 * @author xazhang
 */
class CoordinateDescent(
    coordinates: Seq[(String, Coordinate[_ <: DataSet[_], _ <: Coordinate[_, _]])],
    trainingLossFunctionEvaluator: Evaluator,
    validatingDataAndEvaluatorOption: Option[(RDD[(Long, GameData)], Evaluator)],
    logger: PhotonLogger) {

  /**
   * Run coordinate descent
   *
   * @param numIterations number of iterations
   * @param seed random seed (default: MathConst.RANDOM_SEED)
   * @return trained models
   */
  def run(numIterations: Int, seed: Long = MathConst.RANDOM_SEED): Map[String, Model] = {
    val initializedModelContainer = coordinates.map { case (coordinateId, coordinate) =>
      val initializedModel = coordinate.initializeModel(seed)
      initializedModel match {
        case rddLike: RDDLike => rddLike.setName(s"Initialized model with coordinate id $coordinateId")
              .persistRDD(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)
        case _ =>
      }
      logger.logDebug(s"Summary of model (${initializedModel.getClass}}) initialized for coordinate with " +
          s"ID $coordinateId:\n${initializedModel.toSummaryString}\n")
      (coordinateId, initializedModel)
    }.toMap

    run(numIterations, initializedModelContainer)
  }

  /**
   * Run coordinate descent
   *
   * @param numIterations number of iterations
   * @param modelContainer existing models
   * @return trained models
   */
  def run(numIterations: Int, modelContainer: Map[String, Model]): Map[String, Model] = {

    var updatedModelContainer = modelContainer

    // Initialize the training scores
    var updatedScoresContainer = coordinates.map { case (coordinateId, coordinate) =>
      val updatedScores = coordinate.score(updatedModelContainer(coordinateId))
      updatedScores.setName(s"Initialized training scores with coordinateId $coordinateId")
          .persistRDD(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)
      (coordinateId, updatedScores)
    }.toMap

    // Initialize the validating scores
    var validatingScoresContainerOption = validatingDataAndEvaluatorOption.map { case (validatingData, _) =>
      val validatingScoresContainer = updatedModelContainer.map { case (coordinateId, updatedModel) =>
        val validatingScores = updatedModel.score(validatingData)
        validatingScores.setName(s"Initialized validating scores with coordinateId $coordinateId")
            .persistRDD(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)
        (coordinateId, validatingScores)
      }.toMap
      validatingScoresContainer
    }

    // Initialize the regularization term value
    var regularizationTermValueContainer = coordinates.map { case (coordinateId, coordinate) =>
      (coordinateId, coordinate.computeRegularizationTermValue(updatedModelContainer(coordinateId)))
    }.toMap

    for (iteration <- 0 until numIterations) {
      val iterationStartTime = System.nanoTime()
      logger.logDebug(s"Iteration $iteration of coordinate descent starts...\n")
      coordinates.foreach { case (coordinateId, coordinate) =>

        val coordinateStartTime = System.nanoTime()
        logger.logDebug(s"Start to update coordinate with ID $coordinateId (${coordinate.getClass})")

        // Update the model
        val modelUpdatingStartTime = System.nanoTime()
        val partialScore = updatedScoresContainer.filterKeys(_ != coordinateId).values.reduce(_ + _)
        val oldModel = updatedModelContainer(coordinateId)
        val (updatedModel, optimizationTracker) = coordinate.updateModel(oldModel, partialScore)
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
        updatedModelContainer = updatedModelContainer.updated(coordinateId, updatedModel)

        // Summarize the current progress
        val modelUpdatingElapsedTime = (System.nanoTime() - modelUpdatingStartTime) * 1e-9
        logger.logInfo(s"Finished training the model in coordinate $coordinateId, " +
            s"time elapsed: $modelUpdatingElapsedTime (s).")
        logger.logDebug(s"OptimizationTracker:\n${optimizationTracker.toSummaryString}")
        logger.logDebug(s"Summary of the learned model:\n${updatedModel.toSummaryString}")
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
        logger.logInfo(s"Training objective function value after updating coordinate with id $coordinateId at " +
            s"iteration $iteration is:\n$objectiveFunctionValue")

        // Update the validating score and evaluate the updated model on the validating data
        validatingScoresContainerOption = validatingDataAndEvaluatorOption.map { case (validatingData, evaluator) =>
          val validationStartTime = System.nanoTime()
          var validatingScoresContainer = validatingScoresContainerOption.get
          val validatingScores = updatedModelContainer(coordinateId).score(validatingData)
              .setName(s"Updated validating scores with coordinateId $coordinateId")
              .persistRDD(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)
              .materialize()
          validatingScoresContainer(coordinateId).unpersistRDD()
          validatingScoresContainer = validatingScoresContainer.updated(coordinateId, validatingScores)
          val fullScore = validatingScoresContainer.values.reduce(_ + _)
          val evaluationMetric = evaluator.evaluate(fullScore.scores)
          val validationElapsedTime = (System.nanoTime() - validationStartTime) * 1e-9
          logger.logDebug(s"Finished validating the model, time elapsed: $validationElapsedTime (s).")
          logger.logInfo(s"Evaluation metric after updating coordinateId $coordinateId at iteration $iteration is " +
              s"$evaluationMetric")
          validatingScoresContainer
        }
        val elapsedTime = (System.nanoTime() - coordinateStartTime) * 1e-9
        logger.logInfo(s"Updating coordinate $coordinateId finished, time elapsed: $elapsedTime (s)\n")
      }
      val elapsedTime = (System.nanoTime() - iterationStartTime) * 1e-9
      logger.logInfo(s"Iteration $iteration of coordinate descent finished, time elapsed: $elapsedTime (s)\n\n")
    }

    updatedModelContainer
  }
}
