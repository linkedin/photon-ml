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
package com.linkedin.photon.ml.cli.game

/**
 * Feature params common to GAME training and scoring.
 */
trait FeatureParams {

  /**
   * Input path to the features name-and-term lists.
   */
  var featureNameAndTermSetInputPath: String = ""

  /**
   * A map between the feature shard id and it's corresponding feature section keys in the following format:
   * shardId1:sectionKey1,sectionKey2|shardId2:sectionKey2,sectionKey3.
   */
  var featureShardIdToFeatureSectionKeysMap: Map[String, Set[String]] = Map()

  /**
   * A map between the feature shard id and a boolean variable that decides whether a dummy feature should be added
   * to the corresponding shard in order to learn an intercept, for example,
   * in the following format: shardId1:true|shardId2:false. The default is true for all or unspecified shard ids.
   */
  var featureShardIdToInterceptMap: Map[String, Boolean] = Map()
}
