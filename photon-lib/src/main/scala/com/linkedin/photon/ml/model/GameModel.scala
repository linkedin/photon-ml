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
package com.linkedin.photon.ml.model

import scala.collection.{Map, SortedMap}

import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.TaskType.TaskType
import com.linkedin.photon.ml.data.GameDatum
import com.linkedin.photon.ml.data.scoring.{CoordinateDataScores, ModelDataScores}
import com.linkedin.photon.ml.util.ClassUtils

/**
 * Generalized additive mixed effect (GAME) model.
 *
 * @param gameModels A (modelName -> model) map containing the sub-models that make up the complete GAME model
 */
class GameModel(private val gameModels: Map[String, DatumScoringModel]) extends DatumScoringModel {

  // TODO: This needs to be lazy to be overwritten by anonymous functions without triggering a call to
  // determineModelType. However, for non-anonymous instances of GameModel (i.e. those not created from an existing
  // GameModel) we want this check to run at construction time. That's why modelType is materialized immediately below.
  override lazy val modelType = GameModel.determineModelType(gameModels)
  modelType

  /**
   * Get a sub-model by name.
   *
   * @param name The model name
   * @return An [[Option]] containing the sub-model associated with `name` in the GAME model, or `None` if none exists.
   */
  def getModel(name: String): Option[DatumScoringModel] = gameModels.get(name)

  /**
   * Creates an updated GAME model by updating a named sub-model.
   *
   * @param name The name of the sub-model to update
   * @param model The new model
   * @return A GAME model with updated sub-model `name`
   */
  def updateModel(name: String, model: DatumScoringModel): GameModel = {

    getModel(name).foreach { oldModel =>
      val oldModelClass = ClassUtils.getTrueClass(oldModel)
      val newModelClass = ClassUtils.getTrueClass(model)

      if (!oldModelClass.equals(newModelClass)) {
        throw new UnsupportedOperationException(s"Update model of $oldModelClass to model of $newModelClass is not " +
          s"supported")
      }
    }

    // TODO: The model types don't necessarily match, but checking each time is slow so copy the type for now
    val currType = this.modelType
    new GameModel(gameModels.updated(name, model)) { override lazy val modelType: TaskType = currType }
  }

  /**
   * Convert the GAME model into a (modelName -> model) map representation.
   *
   * @return A (modelName -> model) map representation of this GAME model
   */
  protected[ml] def toMap: Map[String, DatumScoringModel] = gameModels

  /**
   * Convert the GAME model into a (modelName -> model) map representation with a reliable ordering on the keys.
   *
   * @return The (modelName -> model) map representation of this GAME model
   */
  protected[ml] def toSortedMap: SortedMap[String, DatumScoringModel] = SortedMap(gameModels.toSeq: _*)

  /**
   * Compute score, PRIOR to going through any link function, i.e. just compute a dot product of feature values
   * and model coefficients.
   *
   * @param dataPoints The dataset to score (Note that the Long in the RDD is a unique identifier for the paired
   *                   [[GameDatum]] object, referred to in the GAME code as the "unique id")
   * @return The computed scores
   */
  override def score(dataPoints: RDD[(Long, GameDatum)]): ModelDataScores = {
    gameModels.values.map(_.score(dataPoints)).reduce(_ + _)
  }

  override def scoreForCoordinateDescent(dataPoints: RDD[(Long, GameDatum)]): CoordinateDataScores = {
    gameModels.values.map(_.scoreForCoordinateDescent(dataPoints)).reduce(_ + _)
  }

  /**
   * Summarize this GAME model.
   *
   * @return A summary of the object in string representation
   */
  override def toSummaryString: String = {
    gameModels.map { case (name, model) => s"Model name: $name, summary:\n${model.toSummaryString}\n" }.mkString("\n")
  }

  /**
   * Compares two [[GameModel]] objects.
   *
   * @param that Some other object
   * @return True if both models are of the same type, and have the same sub-models with the same names, false otherwise
   */
  override def equals(that: Any): Boolean =
    that match {
      case other: GameModel =>
        (modelType == other.modelType) &&
          this.gameModels.keySet.equals(other.gameModels.keySet) &&
          this.gameModels.forall {
            case (name, model) => other.getModel(name).exists(_.equals(model))
          }

      case _ => false
    }

  /**
   * Returns a hash code value for the object.
   *
   * TODO: Violation of the hashCode() contract
   *
   * @return An [[Int]] hash code
   */
  override def hashCode(): Int = super.hashCode()
}

object GameModel {

  /**
   * Factory method to make code more readable.
   *
   * @param subModels The sub-models that make up this [[GameModel]]
   * @return A new GameModel
   */
  def apply(subModels: (String, DatumScoringModel)*): GameModel = new GameModel(Map(subModels:_*))

  /**
   * Determine the GAME model type: even though a model may have many sub-problems, there is only one loss function type
   * for a given GAME model.
   *
   * @param gameModels A (modelName -> model) map containing the models that make up the complete GAME model
   * @return The GAME model type
   */
  private def determineModelType(gameModels: Map[String, DatumScoringModel]): TaskType = {
    val modelTypes = gameModels.values.map(_.modelType).toSet

    require(modelTypes.size == 1, s"GAME model has multiple model types:\n${modelTypes.mkString(", ")}")

    modelTypes.head
  }
}