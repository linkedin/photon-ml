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

import com.linkedin.photon.ml.{BroadcastLike, RDDLike}
import com.linkedin.photon.ml.data.{KeyValueScore, GameDatum}

/**
 * Representation of the generalized additive mixed effect (GAME) model
 *
 * @param gameModels a (modelName -> model) map representation of the models that consist of the GAME model
 */
class GAMEModel(gameModels: Map[String, Model]) extends Model {

  /**
   * Get the model by name
   * @param name model name
   * @return an option value containing the value associated with model name `name` in the GAME model, or `None`
   *         if none exists.
   */
  def getModel(name: String): Option[Model] = {
    gameModels.get(name)
  }

  /**
   * Creates a updated GAME model obtained by updating it's model with name `name`
   * @param name the name of the model to be updated
   * @param model the model used to update the previous model
   * @return the GAME model with updated model
   */
  def updateModel(name: String, model: Model): GAMEModel = {
    getModel(name).foreach { oldModel =>
      if (!oldModel.getClass.equals(model.getClass)) {
        throw new UnsupportedOperationException(s"Update model of class ${oldModel.getClass} " +
          s"to ${model.getClass} is not supported!")
      }
    }
    new GAMEModel(gameModels.updated(name, model))
  }

  /**
   * Convert the GAME model into a (modelName -> model) map representation
   * @return the (modelName -> model) map representation of the models
   */
  protected[ml] def toMap: Map[String, Model] = {
    gameModels
  }

  /**
   * Persist each model with the specified storage level if it's a RDD
   * @param storageLevel the storage level
   * @return self with all RDD like models persisted
   */
  def persist(storageLevel: StorageLevel): this.type = {
    gameModels.values.foreach {
      case rddLike: RDDLike => rddLike.persistRDD(storageLevel)
      case _ =>
    }
    this
  }

  def unpersist: this.type = {
    gameModels.values.foreach {
      case rddLike: RDDLike => rddLike.unpersistRDD()
      case broadcastLike: BroadcastLike => broadcastLike.unpersistBroadcast()
      case _ =>
    }
    this
  }

  override def score(dataPoints: RDD[(Long, GameDatum)]): KeyValueScore = {
    gameModels.values.map(_.score(dataPoints)).reduce(_ + _)
  }

  override def toSummaryString: String = {
    gameModels.map { case (name, model) => s"Model name: $name, summary:\n${model.toSummaryString}\n" }.mkString("\n")
  }

  override def equals(that: Any): Boolean = {
    that match {
      case other: GAMEModel => gameModels.forall { case (name, model) =>
        other.getModel(name).isDefined && other.getModel(name).get.equals(model)
      }
      case _ => false
    }
  }

  override def hashCode(): Int = {
    super.hashCode()
  }
}
