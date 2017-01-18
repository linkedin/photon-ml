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

import scala.collection.Map

import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import com.linkedin.photon.ml.data.{GameDatum, KeyValueScore}
import com.linkedin.photon.ml.TaskType.TaskType
import com.linkedin.photon.ml.{BroadcastLike, RDDLike, TaskType}

/**
 * Generalized additive mixed effect (GAME) model.
 *
 * @param gameModels A (modelName -> model) map containing the models that make up the complete GAME model
 * @param taskType the type of model. Even though there are many sub-problems, there is only one loss function type
 *                 for a given GameModel.
 */
class GAMEModel(gameModels: Map[String, DatumScoringModel], val taskType: TaskType = TaskType.NONE)
    extends DatumScoringModel {

  checkInvariants()

  /**
   * This method checks GameModel's invariants.
   *
   * Invariant #1: all sub-models are of type taskType
   */
  def checkInvariants(): Unit = {

    val taskType = this.taskType // so Spark can close and serialize the lambda in the foreach below

    gameModels.foreach {
      case (sectionName, datumScoringModel: FixedEffectModel) =>
        require(TaskType.matches(datumScoringModel.model, taskType),
          s"Expecting ${taskType.toString} for $sectionName but got ${TaskType.name(datumScoringModel.model)}")

      case (_, datumScoringModel: RandomEffectModel) =>
        datumScoringModel.modelsRDD.foreach {
          case (modelName, model) => require(TaskType.matches(model, taskType),
            s"Expecting ${taskType.toString} for $modelName but got ${TaskType.name(model)}")
        }
    }
  }

  /**
   * Get a (sub-) model by name
   *
   * @param name The model name
   * @return An option value containing the value associated with model `name` in the GAME model,
   *         or `None` if none exists.
   */
  def getModel(name: String): Option[DatumScoringModel] = gameModels.get(name)


  /**
   * Creates an updated GAME model by updating (sub-) model `name`
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
    new GAMEModel(gameModels.updated(name, model), taskType)
  }

  /**
   * Convert the GAME model into a (modelName -> model) map representation
   *
   * @return The (modelName -> model) map representation of the models
   */
  protected[ml] def toMap: Map[String, DatumScoringModel] = gameModels

  /**
   * Persist each model with the specified storage level if it's a RDD
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
   * Unpersist each model if it's a RDD
   *
   * @return Myself with all RDD like models unpersisted
   */
  def unpersist: this.type = {

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

  override def toSummaryString: String = {
    gameModels.map { case (name, model) => s"Model name: $name, summary:\n${model.toSummaryString}\n" }.mkString("\n")
  }

  override def equals(that: Any): Boolean = {

    that match {
      case other: GAMEModel =>
        (taskType == other.taskType) && gameModels.forall {
          case (name, model) =>
            other.getModel(name).isDefined && other.getModel(name).get.equals(model)
        }
      case _ => false
    }
  }

  // TODO: Violation of the hashCode() contract
  override def hashCode(): Int = {
    super.hashCode()
  }
}

object GAMEModel {

  /**
   * Factory method to make code more readable.
   *
   * @param taskType The model type for this GameModel
   * @param sections The sections that make up this GameModel
   * @return A new instance of GAMEModel
   */
  def apply(taskType: TaskType, sections: (String, DatumScoringModel)*): GAMEModel =
    new GAMEModel(Map(sections:_*), taskType)
}