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

import com.linkedin.photon.ml.data.{KeyValueScore, GameDatum}
import com.linkedin.photon.ml.util.Summarizable
import org.apache.spark.rdd.RDD

/**
  * Interface for the implementation of a GAME model
  */
protected[ml] trait DatumScoringModel extends Summarizable {
  /**
    * Compute the score for the GAME data set.
    *
    * @param dataPoints The dataset, which is a RDD consists of the (global Id, GameDatum) pairs. Note that the Long in
    *                   the RDD above is a unique identifier for which GenericRecord the GameData object was created,
    *                   referred to in the GAME code as the "global ID".
    * @return The score
    */
  def score(dataPoints: RDD[(Long, GameDatum)]): KeyValueScore
}
