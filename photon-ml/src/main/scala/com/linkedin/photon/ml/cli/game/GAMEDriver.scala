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
package com.linkedin.photon.ml.cli.game

import com.linkedin.photon.ml.util._
import com.linkedin.photon.ml.avro.data.NameAndTermFeatureSetContainer

import org.apache.spark.SparkContext

/**
  * Contains common functions for GAME training and scoring drivers.
  */
class GAMEDriver(
    params: FeatureParams with PalDBIndexMapParams,
    sparkContext: SparkContext,
    logger: PhotonLogger) {

  import params._

  protected val hadoopConfiguration = sparkContext.hadoopConfiguration

  /**
    * Builds feature key to index map loaders according to configuration
    *
    * @return A map of shard id to feature map loader
    * @deprecated This function will be removed in the next major version.
    */
  protected[game] def prepareFeatureMapsDefault(): Map[String, IndexMapLoader] = {
    val allFeatureSectionKeys = featureShardIdToFeatureSectionKeysMap.values.reduce(_ ++ _)
    val nameAndTermFeatureSetContainer = NameAndTermFeatureSetContainer.readNameAndTermFeatureSetContainerFromTextFiles(
      featureNameAndTermSetInputPath, allFeatureSectionKeys, hadoopConfiguration)

    val featureShardIdToFeatureMapLoader =
      featureShardIdToFeatureSectionKeysMap.map { case (shardId, featureSectionKeys) =>
        val featureMap = nameAndTermFeatureSetContainer
          .getFeatureNameAndTermToIndexMap(featureSectionKeys, featureShardIdToInterceptMap.getOrElse(shardId, true))
          .map { case (k, v) => Utils.getFeatureKey(k.name, k.term) -> v }
          .toMap

        val indexMapLoader = new DefaultIndexMapLoader(featureMap)
        indexMapLoader.prepare(sparkContext, null, shardId)
        (shardId, indexMapLoader)
      }
    featureShardIdToFeatureMapLoader.foreach { case (shardId, featureMapLoader) =>
      logger.debug(s"Feature shard ID: $shardId, number of features: ${featureMapLoader.indexMapForDriver.size}")
    }
    featureShardIdToFeatureMapLoader
  }

  /**
    * Builds PalDB off-heap feature name-and-term to index map loaders according to configuration
    *
    * @return A map of shard id to feature map
    */
  protected[game] def prepareFeatureMapsPalDB(): Map[String, IndexMapLoader] = {
    featureShardIdToFeatureSectionKeysMap.map { case (shardId, featureSections) => {
      val indexMapLoader = new PalDBIndexMapLoader
      indexMapLoader.prepare(sparkContext, params, shardId)
      (shardId, indexMapLoader)
    }}
  }

  /**
    * Builds feature name-and-term to index maps according to configuration
    *
    * @return A map of shard id to feature map
    */
  protected[game] def prepareFeatureMaps(): Map[String, IndexMapLoader] = {
    params.offHeapIndexMapDir match {
      // If an off-heap map path is specified, use the paldb loader
      case Some(_) => prepareFeatureMapsPalDB()

      // Otherwise, fall back to the default loader
      case _ => prepareFeatureMapsDefault()
    }
  }
}
