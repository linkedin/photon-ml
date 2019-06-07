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
package com.linkedin.photon.ml.cli.game.scoring

/**
 * A compact representation of the scored item.
 *
 * @param predictionScore The prediction score
 * @param label An optional label of the score
 * @param weight An optional weight of the score
 * @param idTagToValueMap A map of ID tag to ID for this data point. An ID tag is a column or metadata field containing
 *                        IDs used to group or uniquely identify samples. Examples of ID tags that may be stored as keys
 *                        in this map are:
 *
 *                        (i) ID tags used to build random effect models (e.g. userId, jobId);
 *                        (ii) ID tags used to compute evaluation metrics like precision@k (e.g. documentId, queryId);
 *                        (iii) ID tags used to uniquely identify training records (e.g. uid)
 */
case class ScoredItem(
    predictionScore: Double,
    label: Option[Double],
    weight: Option[Double],
    idTagToValueMap: Map[String, String])
