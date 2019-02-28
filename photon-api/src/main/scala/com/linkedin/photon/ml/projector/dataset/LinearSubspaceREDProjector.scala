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
package com.linkedin.photon.ml.projector.dataset

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import com.linkedin.photon.ml.Types.REId
import com.linkedin.photon.ml.data.{LabeledPoint, RandomEffectDataset}
import com.linkedin.photon.ml.projector.vector.LinearSubspaceProjector
import com.linkedin.photon.ml.spark.RDDLike

/**
 * Project each per-entity dataset stored in a [[RandomEffectDataset]] between spaces, where each entity has a unique
 * projected space which is a linear subspace of the shared original space.
 *
 * For an example use case, see the [[LinearSubspaceProjector]] documentation.
 *
 * @param linearSubspaceProjectorsRDD A [[RDD]] of (entity ID, [[LinearSubspaceProjector]]) pairs
 */
protected[ml] class LinearSubspaceREDProjector(linearSubspaceProjectorsRDD: RDD[(REId, LinearSubspaceProjector)])
  extends RandomEffectDatasetProjector with RDDLike {

  /**
   * Project a [[RandomEffectDataset]] from the original space to the projected space.
   *
   * @param randomEffectDataset The [[RandomEffectDataset]] in the original space
   * @return The same [[RandomEffectDataset]] in the projected space
   */
  override def projectForward(randomEffectDataset: RandomEffectDataset): RandomEffectDataset = {

    val activeData = randomEffectDataset.activeData
    val passiveData = randomEffectDataset.passiveData
    val passiveDataREIds = randomEffectDataset.passiveDataREIds

    // Make sure the activeData retains its partitioner, especially when the partitioner of featureMaps is
    // not the same as that of activeData
    val projectedActiveData = activeData
      .join(linearSubspaceProjectorsRDD, activeData.partitioner.get)
      .mapValues { case (localDataset, projector) => localDataset.projectForward(projector) }

    val projectorsForPassiveData = linearSubspaceProjectorsRDD
        .filter { case (randomEffectId, _) =>
          passiveDataREIds.value.contains(randomEffectId)
        }
        .collectAsMap()
    val projectedPassiveData = passiveData.mapValues { case (shardId, LabeledPoint(response, features, offset, weight)) =>
      val projector = projectorsForPassiveData(shardId)
      val projectedFeatures = projector.projectForward(features)

      (shardId, LabeledPoint(response, projectedFeatures, offset, weight))
    }

    randomEffectDataset.update(projectedActiveData, projectedPassiveData)
  }

  /**
   * Project a [[RandomEffectDataset]] from the projected space to the original space.
   *
   * @param randomEffectDataset The [[RandomEffectDataset]] in the projected space
   * @return The same [[RandomEffectDataset]] in the original space
   */
  override def projectBackward(randomEffectDataset: RandomEffectDataset): RandomEffectDataset = {

    val activeData = randomEffectDataset.activeData
    val passiveData = randomEffectDataset.passiveData
    val passiveDataREIds = randomEffectDataset.passiveDataREIds

    // Make sure the activeData retains its partitioner, especially when the partitioner of featureMaps is
    // not the same as that of activeData
    val projectedActiveData = activeData
      .join(linearSubspaceProjectorsRDD, randomEffectDataset.randomEffectIdPartitioner)
      .mapValues { case (localDataset, projector) => localDataset.projectBackward(projector) }

    val projectorsForPassiveData = linearSubspaceProjectorsRDD
      .filter { case (randomEffectId, _) =>
        passiveDataREIds.value.contains(randomEffectId)
      }
      .collectAsMap()
    val projectedPassiveData = passiveData.mapValues { case (shardId, LabeledPoint(response, features, offset, weight)) =>
      val projector = projectorsForPassiveData(shardId)
      val projectedFeatures = projector.projectBackward(features)

      (shardId, LabeledPoint(response, projectedFeatures, offset, weight))
    }

    randomEffectDataset.update(projectedActiveData, projectedPassiveData)
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
  override def setName(name: String): LinearSubspaceREDProjector = {

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
  override def persistRDD(storageLevel: StorageLevel): LinearSubspaceREDProjector = {

    if (!linearSubspaceProjectorsRDD.getStorageLevel.isValid) linearSubspaceProjectorsRDD.persist(storageLevel)

    this
  }

  /**
   * Mark all blocks of [[linearSubspaceProjectorsRDD]] for removal from memory and disk.
   *
   * @return This object with [[linearSubspaceProjectorsRDD]] marked for removal
   */
  override def unpersistRDD(): LinearSubspaceREDProjector = {

    if (linearSubspaceProjectorsRDD.getStorageLevel.isValid) linearSubspaceProjectorsRDD.unpersist()

    this
  }

  /**
   * Materialize [[linearSubspaceProjectorsRDD]] (Spark [[RDD]]s are lazy evaluated: this method forces evaluation).
   *
   * @return This object with [[linearSubspaceProjectorsRDD]] materialized
   */
  override def materialize(): LinearSubspaceREDProjector = {

    linearSubspaceProjectorsRDD.count()

    this
  }
}
