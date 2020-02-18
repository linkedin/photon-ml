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

import scala.collection.SortedMap

import org.apache.spark.sql.DataFrame

import com.linkedin.photon.ml.TaskType.TaskType
import com.linkedin.photon.ml.Types.CoordinateId
import com.linkedin.photon.ml.util.ClassUtils

/**
 * Generalized additive mixed effect (GAME) model.
 *
 * @param gameModels A (modelName -> model) map containing the sub-models that make up the complete GAME model
 */
class GameModel (private val gameModels: Map[CoordinateId, DatumScoringModel]) extends DatumScoringModel {

  // The model type should be consistent at construction time. However, copies of this object shouldn't need to call the
  // check again. Thus the value is lazy, so that anonymous classes can overwrite it without triggering a call to
  // determineModelType, but it's called immediately so that it's evaluated at construction time.
  override lazy val modelType: TaskType = GameModel.determineModelType(gameModels)
  modelType

  /**
   * Get a sub-model by name.
   *
   * @param name The model name
   * @return An [[Option]] containing the sub-model associated with `name` in the GAME model, or `None` if none exists.
   */
  def getModel(name: CoordinateId): Option[DatumScoringModel] = gameModels.get(name)

  /**
   * Creates an updated GAME model by updating a named sub-model.
   *
   * @param name The name of the sub-model to update
   * @param model The new model
   * @return A GAME model with updated sub-model `name`
   */
  def updateModel(name: CoordinateId, model: DatumScoringModel): GameModel = {

    val oldModel = gameModels(name)
    val oldModelClass = ClassUtils.getTrueClass(oldModel)
    val newModelClass = ClassUtils.getTrueClass(model)

    require(
      oldModelClass.equals(newModelClass),
      s"$name: Update model of class $oldModelClass to model of class $newModelClass is not supported")
    require(
      oldModel.modelType == model.modelType,
      s"$name: Updated model type ${model.modelType} does not match current type ${oldModel.modelType}")

    // Since all component models must have the same type at construction time, and the updated model has the same type
    // as the previous model, therefore the model type must still match and thus the value of the new model can be
    // explicitly set
    val currType = modelType
    new GameModel(gameModels.updated(name, model)) {
      override lazy val modelType: TaskType = currType
    }
  }

  /**
   * Convert the GAME model into a (modelName -> model) map representation.
   *
   * @return A (modelName -> model) map representation of this GAME model
   */
  def toMap: Map[CoordinateId, DatumScoringModel] = gameModels

  /**
   * Convert the GAME model into a (modelName -> model) map representation with a reliable ordering on the keys.
   *
   * @return The (modelName -> model) map representation of this GAME model
   */
  def toSortedMap: SortedMap[CoordinateId, DatumScoringModel] = SortedMap(gameModels.toSeq: _*)

  /**
   * Compute score, PRIOR to going through any link function, i.e. just compute a dot product of feature values
   * and model coefficients.
   *
   * @param dataPoints The dataset to score
   * @return The computed scores
   */
  override def score(dataPoints: DataFrame): DataFrame =
    gameModels.values.map(_.score(dataPoints)).reduce(_ + _)

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
  def apply(subModels: (CoordinateId, DatumScoringModel)*): GameModel = new GameModel(Map(subModels:_*))

  /**
   * Determine the GAME model type: even though a model may have many sub-problems, there is only one loss function type
   * for a given GAME model.
   *
   * @param gameModels A (modelName -> model) map containing the models that make up the complete GAME model
   * @return The GAME model type
   */
  private def determineModelType(gameModels: Map[CoordinateId, DatumScoringModel]): TaskType = {
    val modelTypes = gameModels.values.map(_.modelType).toSet

    require(modelTypes.size == 1, s"GAME model has multiple model types:\n${modelTypes.mkString(", ")}")

    modelTypes.head
  }
}
