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
package com.linkedin.photon.ml.cli.game.scoring

import scala.collection.Map

/**
 * A compact representation of the scored item.
 *
 * @param predictionScore The prediction score
 * @param label An optional label of the score
 * @param weight An optional weight of the score
 * @param idTypeToValueMap The id type to value map that holds different types of ids associated with this data
 *                         point, e.g. Map("userId" -> "1234", "itemId" -> "abcd").
 */
case class ScoredItem(
    predictionScore: Double,
    label: Option[Double],
    weight: Option[Double],
    idTypeToValueMap: Map[String, String])
