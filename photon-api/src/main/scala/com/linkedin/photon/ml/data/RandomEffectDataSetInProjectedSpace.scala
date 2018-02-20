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
package com.linkedin.photon.ml.data

import org.apache.spark.storage.StorageLevel

import com.linkedin.photon.ml.projector.{ProjectorType, RandomEffectProjector}
import com.linkedin.photon.ml.spark.RDDLike

/**
 * Dataset implementation for random effect datasets in projected space.
 *
 * @param randomEffectDataSetInProjectedSpace Input random effect dataset
 * @param randomEffectProjector The random effect projector
 */
class RandomEffectDataSetInProjectedSpace(
    val randomEffectDataSetInProjectedSpace: RandomEffectDataSet,
    val randomEffectProjector: RandomEffectProjector)
  extends RandomEffectDataSet(
    randomEffectDataSetInProjectedSpace.activeData,
    randomEffectDataSetInProjectedSpace.uniqueIdToRandomEffectIds,
    randomEffectDataSetInProjectedSpace.passiveDataOption,
    randomEffectDataSetInProjectedSpace.passiveDataRandomEffectIdsOption,
    randomEffectDataSetInProjectedSpace.randomEffectType,
    randomEffectDataSetInProjectedSpace.featureShardId) {

  /**
   *
   * @param name The parent name for all RDDs in this class
   * @return This object with all its RDDs' name assigned
   */
  override def setName(name: String): this.type = {
    super.setName(name)
    randomEffectProjector match {
      case rddLike: RDDLike => rddLike.setName(s"$name: projector ${randomEffectProjector.getClass}")
      case _ =>
    }
    this
  }

  /**
   *
   * @param storageLevel The storage level
   * @return This object with all its RDDs' storage level set
   */
  override def persistRDD(storageLevel: StorageLevel): this.type = {
    super.persistRDD(storageLevel)
    randomEffectProjector match {
      case rddLike: RDDLike => rddLike.persistRDD(storageLevel)
      case _ =>
    }
    this
  }

  /**
   *
   * @return This object with all its RDDs unpersisted
   */
  override def unpersistRDD(): this.type = {
    // TODO: Projection needs to be refactored in general - the RandomEffectProjector gets passed around between classes
    // and has no one owner
//    randomEffectProjector match {
//      case rddLike: RDDLike => rddLike.unpersistRDD()
//      case _ =>
//    }
    super.unpersistRDD()
    this
  }

  /**
   *
   * @return This object with all its RDDs materialized
   */
  override def materialize(): this.type = {
    super.materialize()
    randomEffectProjector match {
      case rddLike: RDDLike => rddLike.materialize()
      case _ =>
    }
    this
  }
}

object RandomEffectDataSetInProjectedSpace {
  /**
   * Build an instance of a random effect dataset in projected space with the given projector type.
   *
   * @param randomEffectDataSet The input dataset
   * @param projectorType The projector type
   * @return A new dataset projected with the given projector
   */
  def buildWithProjectorType(
      randomEffectDataSet: RandomEffectDataSet,
      projectorType: ProjectorType): RandomEffectDataSetInProjectedSpace = {

    val randomEffectProjector = RandomEffectProjector.build(randomEffectDataSet, projectorType)
    val projectedRandomEffectDataSet = randomEffectProjector.projectRandomEffectDataSet(randomEffectDataSet)
    new RandomEffectDataSetInProjectedSpace(projectedRandomEffectDataSet, randomEffectProjector)
  }
}
