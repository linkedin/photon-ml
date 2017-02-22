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
package com.linkedin.photon.ml.projector

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import com.linkedin.photon.ml.data.{LabeledPoint, RandomEffectDataSet}
import com.linkedin.photon.ml.model.Coefficients
import com.linkedin.photon.ml.spark.RDDLike
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel

/**
 * A class that holds the projectors for a sharded data set.
 *
 * @param indexMapProjectorRDD The projectors
 */
protected[ml] class IndexMapProjectorRDD private (indexMapProjectorRDD: RDD[(String, IndexMapProjector)])
  extends RandomEffectProjector
  with RDDLike {

  /**
   *
   * @param randomEffectDataSet The input sharded data set in the original space
   * @return The sharded data set in the projected space
   */
  override def projectRandomEffectDataSet(randomEffectDataSet: RandomEffectDataSet): RandomEffectDataSet = {
    val activeData = randomEffectDataSet.activeData
    val passiveDataOption = randomEffectDataSet.passiveDataOption
    val passiveDataRandomEffectIdsOption = randomEffectDataSet.passiveDataRandomEffectIdsOption
    val projectedActiveData =
      activeData
        // Make sure the activeData retains its partitioner, especially when the partitioner of featureMaps is
        // not the same as that of activeData
        .join(indexMapProjectorRDD, activeData.partitioner.get)
        .mapValues { case (localDataSet, projector) => localDataSet.projectFeatures(projector) }

    val projectedPassiveData =
      if (passiveDataOption.isDefined) {
        val passiveDataRandomEffectIds = passiveDataRandomEffectIdsOption.get
        val projectorsForPassiveData = indexMapProjectorRDD.filter { case (randomEffectId, _) =>
          passiveDataRandomEffectIds.value.contains(randomEffectId)
        }.collectAsMap()

        val projectorsForPassiveDataBroadcast = passiveDataOption.get.sparkContext.broadcast(projectorsForPassiveData)
        val result = passiveDataOption.map {
          _.mapValues { case (shardId, LabeledPoint(response, features, offset, weight)) =>
            val projector = projectorsForPassiveDataBroadcast.value(shardId)
            val projectedFeatures = projector.projectFeatures(features)

            (shardId, LabeledPoint(response, projectedFeatures, offset, weight))
          }
        }

        projectorsForPassiveDataBroadcast.unpersist()
        result
      } else {
        None
      }

    randomEffectDataSet.update(projectedActiveData, projectedPassiveData)
  }

  /**
   *
   * @param coefficientsRDD
   * @return The [[RDD]] of [[Coefficients]] in the original space
   */
  override def projectCoefficientsRDD(
      coefficientsRDD: RDD[(String, GeneralizedLinearModel)]): RDD[(String, GeneralizedLinearModel)] = {

    coefficientsRDD
      .join(indexMapProjectorRDD)
      .mapValues { case (model, projector) =>
        val oldCoefficients = model.coefficients
        model.updateCoefficients(
          Coefficients(
            projector.projectCoefficients(oldCoefficients.means),
            oldCoefficients.variancesOption.map(projector.projectCoefficients)))
      }
  }

  /**
   *
   * @return The Spark context
   */
  override def sparkContext: SparkContext = {
    indexMapProjectorRDD.sparkContext
  }

  /**
   *
   * @param name The parent name for all RDDs in this class
   * @return This object with all its RDDs' name assigned
   */
  override def setName(name: String): this.type = {
    indexMapProjectorRDD.setName(name)
    this
  }

  /**
   *
   * @param storageLevel The storage level
   * @return This object with all its RDDs' storage level set
   */
  override def persistRDD(storageLevel: StorageLevel): this.type = {
    if (!indexMapProjectorRDD.getStorageLevel.isValid) indexMapProjectorRDD.persist(storageLevel)
    this
  }

  /**
   *
   * @return This object with all its RDDs unpersisted
   */
  override def unpersistRDD(): this.type = {
    if (indexMapProjectorRDD.getStorageLevel.isValid) indexMapProjectorRDD.unpersist()
    this
  }

  /**
   *
   * @return This object with all its RDDs materialized
   */
  override def materialize(): this.type = {
    indexMapProjectorRDD.count()
    this
  }
}

object IndexMapProjectorRDD {
  /**
   * Generate index map based RDD projectors.
   *
   * @param randomEffectDataSet The input random effect data set
   * @return The generated index map based RDD projectors
   */
  protected[ml] def buildIndexMapProjector(randomEffectDataSet: RandomEffectDataSet): IndexMapProjectorRDD = {
    val indexMapProjectors = randomEffectDataSet.activeData.mapValues(localDataSet =>
      IndexMapProjector.buildIndexMapProjector(localDataSet.dataPoints.map(_._2.features))
    )
    new IndexMapProjectorRDD(indexMapProjectors)
  }
}
