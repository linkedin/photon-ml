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
package com.linkedin.photon.ml.model

import scala.collection.{Map, SortedMap}

import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import com.linkedin.photon.ml.TaskType.TaskType
import com.linkedin.photon.ml.data.{GameDatum, KeyValueScore}
import com.linkedin.photon.ml.spark.{BroadcastLike, RDDLike}

/**
 * Generalized additive mixed effect (GAME) model.
 *
 * @param gameModels A (modelName -> model) map containing the models that make up the complete GAME model
 */
class GAMEModel(gameModels: Map[String, DatumScoringModel]) extends DatumScoringModel {

  val modelType = GAMEModel.determineModelType(gameModels)

  /**
   * Get a (sub-)model by name.
   *
   * @param name The model name
   * @return An option value containing the value associated with model `name` in the GAME model,
   *         or `None` if none exists.
   */
  def getModel(name: String): Option[DatumScoringModel] = gameModels.get(name)

  /**
   * Creates an updated GAME model by updating (sub-)model `name`.
   *
   * @param name The name of the model to be updated
   * @param model The new model
   * @return The GAME model with the updated model
   */
  def updateModel(name: String, model: DatumScoringModel): GAMEModel = {
    getModel(name).foreach { oldModel =>
      if (!oldModel.getClass.equals(model.getClass)) {
        throw new UnsupportedOperationException(s"Update model of class ${oldModel.getClass} " +
          s"to ${model.getClass} is not supported!")
      }
    }

    // TODO: The model types don't necessarily match, but checking each time is slow so copy the type for now
    new GAMEModel(gameModels.updated(name, model)) { override val modelType: TaskType = this.modelType }
  }

  /**
   * Convert the GAME model into a (modelName -> model) map representation.
   *
   * @return The (modelName -> model) map representation of the models
   */
  protected[ml] def toMap: Map[String, DatumScoringModel] = gameModels

  /**
    * Convert the GAME model into a (modelName -> model) map representation that has a reliable order on keys
    *
    * @return The (modelName -> model) map representation of the models
    */
  protected[ml] def toSortedMap: SortedMap[String, DatumScoringModel] = SortedMap(gameModels.toSeq: _*)

  /**
   * Persist each model with the specified storage level if it's a RDD.
   *
   * @param storageLevel The storage level
   * @return Myself with all RDD like models persisted
   */
  def persist(storageLevel: StorageLevel): this.type = {
    gameModels.values.foreach {
      case rddLike: RDDLike => rddLike.persistRDD(storageLevel)
      case _ =>
    }

    this
  }

  /**
   * Unpersist each model if it's a RDD.
   *
   * @return Myself with all RDD like models unpersisted
   */
  def unpersist(): this.type = {
    gameModels.values.foreach {
      case rddLike: RDDLike => rddLike.unpersistRDD()
      case broadcastLike: BroadcastLike => broadcastLike.unpersistBroadcast()
      case _ =>
    }

    this
  }

  /**
   * Compute score, PRIOR to going through any link function, i.e. just compute a dot product of feature values
   * and model coefficients.
   *
   * @param dataPoints The dataset, which is a RDD consists of the (unique id, GameDatum) pairs. Note that the Long in
   *                   the RDD above is a unique identifier for which GenericRecord the GameData object was created,
   *                   referred to in the GAME code as the "unique id".
   * @return The score
   */
  override def score(dataPoints: RDD[(Long, GameDatum)]): KeyValueScore = {
    gameModels.values.map(_.score(dataPoints)).reduce(_ + _)
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
   *
   * @param that
   * @return
   */
  override def equals(that: Any): Boolean = {
    that match {
      case other: GAMEModel =>
        (modelType == other.modelType) && gameModels.forall {
          case (name, model) =>
            other.getModel(name).isDefined && other.getModel(name).get.equals(model)
        }
      case _ => false
    }
  }

  /**
   *
   * @return
   */
  // TODO: Violation of the hashCode() contract
  override def hashCode(): Int = {
    super.hashCode()
  }
}

object GAMEModel {

  /**
   * Factory method to make code more readable.
   *
   * @param sections The sections that make up this GameModel
   * @return A new instance of GAMEModel
   */
  def apply(sections: (String, DatumScoringModel)*): GAMEModel = new GAMEModel(Map(sections:_*))

  /**
   * Determine the GAME model type: even though a model may have many sub-problems, there is only one loss function type
   * for a given GAME model.
   *
   * @param gameModels A (modelName -> model) map containing the models that make up the complete GAME model.
   * @return The GAME model type.
   */
  protected def determineModelType(gameModels: Map[String, DatumScoringModel]): TaskType = {
    val modelTypes = gameModels.values.map(_.modelType).toSet

    require(modelTypes.size == 1, s"GAME model has multiple model types:\n${modelTypes.mkString(", ")}")

    modelTypes.head
  }
}