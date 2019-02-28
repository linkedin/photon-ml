/*
 * Copyright 2019 LinkedIn Corp. All rights reserved.
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
package com.linkedin.photon.ml.projector.model

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import com.linkedin.photon.ml.Types.REId
import com.linkedin.photon.ml.model.{Coefficients, RandomEffectModel}
import com.linkedin.photon.ml.projector.vector.LinearSubspaceProjector
import com.linkedin.photon.ml.spark.RDDLike

/**
 * Project each component model of a [[RandomEffectModel]] between spaces, where each component model has a unique
 * projected space which is a linear subspace of the shared original space.
 *
 * For an example use case, see the [[LinearSubspaceProjector]] documentation.
 *
 * @param linearSubspaceProjectorsRDD A [[RDD]] of (entity ID, [[LinearSubspaceProjector]]) pairs
 */
class LinearSubspaceREMProjector(linearSubspaceProjectorsRDD: RDD[(REId, LinearSubspaceProjector)])
  extends RandomEffectModelProjector with RDDLike {

  /**
   * Project a [[RandomEffectModel]] from the original space to the projected space.
   *
   * @param randomEffectModel The [[RandomEffectModel]] in the original space
   * @return The same [[RandomEffectModel]] in the projected space
   */
  override def projectForward(randomEffectModel: RandomEffectModel): RandomEffectModel = {

    // Left join the models to projectors for cases where we have a prior model but no new model (and hence no
    // projectors)
    val newModels = randomEffectModel
      .modelsRDD
      .leftOuterJoin(linearSubspaceProjectorsRDD)
      .mapValues { case (model, projectorOpt) =>
        projectorOpt
          .map { projector =>
            val oldCoefficients = model.coefficients
            val newCoefficients = Coefficients(
              projector.projectForward(oldCoefficients.means),
              oldCoefficients.variancesOption.map(projector.projectForward))

            model.updateCoefficients(newCoefficients)
          }
          .getOrElse(model)
      }

    randomEffectModel.update(newModels)
  }

  /**
   * Project a [[RandomEffectModel]] from the projected space to the original space.
   *
   * @param randomEffectModel The [[RandomEffectModel]] in the projected space
   * @return The same [[RandomEffectModel]] in the original space
   */
  override def projectBackward(randomEffectModel: RandomEffectModel): RandomEffectModel = {

    // Left join the models to projectors for cases where we have a prior model but no new model (and hence no
    // projectors)
    val newModels = randomEffectModel
      .modelsRDD
      .leftOuterJoin(linearSubspaceProjectorsRDD)
      .mapValues { case (model, projectorOpt) =>
        projectorOpt
          .map { projector =>
            val oldCoefficients = model.coefficients
            val newCoefficients = Coefficients(
              projector.projectBackward(oldCoefficients.means),
              oldCoefficients.variancesOption.map(projector.projectBackward))

            model.updateCoefficients(newCoefficients)
          }
          .getOrElse(model)
      }

    randomEffectModel.update(newModels)
  }

  /**
   * Get the Spark context.
   *
   * @return The Spark context
   */
  override def sparkContext: SparkContext = linearSubspaceProjectorsRDD.sparkContext

  /**
   * Assign a name to [[linearSubspaceProjectorsRDD]].
   *
   * @note Not used to reference models in the logic of photon-ml, only used for logging.
   *
   * @param name The parent name for all [[RDD]]s in this class
   * @return This object with the name of [[linearSubspaceProjectorsRDD]] assigned
   */
  override def setName(name: String): LinearSubspaceREMProjector = {

    linearSubspaceProjectorsRDD.setName(name)

    this
  }

  /**
   * Set the storage level of [[linearSubspaceProjectorsRDD]], and persist its values across the cluster the first time they
   * are computed.
   *
   * @param storageLevel The storage level
   * @return This object with the storage level of [[linearSubspaceProjectorsRDD]] set
   */
  override def persistRDD(storageLevel: StorageLevel): LinearSubspaceREMProjector = {

    if (!linearSubspaceProjectorsRDD.getStorageLevel.isValid) linearSubspaceProjectorsRDD.persist(storageLevel)

    this
  }

  /**
   * Mark all blocks of [[linearSubspaceProjectorsRDD]] for removal from memory and disk.
   *
   * @return This object with [[linearSubspaceProjectorsRDD]] marked for removal
   */
  override def unpersistRDD(): LinearSubspaceREMProjector = {

    if (linearSubspaceProjectorsRDD.getStorageLevel.isValid) linearSubspaceProjectorsRDD.unpersist()

    this
  }

  /**
   * Materialize [[linearSubspaceProjectorsRDD]] (Spark [[RDD]]s are lazy evaluated: this method forces evaluation).
   *
   * @return This object with [[linearSubspaceProjectorsRDD]] materialized
   */
  override def materialize(): LinearSubspaceREMProjector = {

    linearSubspaceProjectorsRDD.count()

    this
  }
}
