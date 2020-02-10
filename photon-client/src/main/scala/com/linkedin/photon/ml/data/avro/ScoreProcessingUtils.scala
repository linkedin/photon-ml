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
package com.linkedin.photon.ml.data.avro

import scala.collection.JavaConverters._

import org.apache.spark.rdd.RDD

import com.linkedin.photon.avro.generated.ScoringResultAvro
import com.linkedin.photon.ml.cli.game.scoring.ScoredItem
import com.linkedin.photon.ml.data.InputColumnsNames

/**
 * Some basic functions to read/write scores computed with GAME model from/to HDFS. The current implementation assumes
 * the scores are stored using Avro format.
 */
object ScoreProcessingUtils {

  val DEFAULT_MODEL_ID = "N/A"

  /**
   * Save the scored items of type [[ScoredItem]] to the given output directory on HDFS.
   *
   * @param scoredItems An [[RDD]] of scored items of type [[ScoredItem]]
   * @param modelId The model's id that used to compute the scores
   * @param outputDir The given output directory
   */
  protected[ml] def saveScoredItemsToHDFS(
      scoredItems: RDD[ScoredItem],
      outputDir: String,
      modelId: Option[String],
      columnsNames: InputColumnsNames): Unit = {

    val scoringResultAvros = scoredItems.map { case ScoredItem(predictionScore, labelOpt, weightOpt, ids) =>
      val metaDataMap = collection.mutable.Map(ids.toMap[CharSequence, CharSequence].toSeq: _*).asJava
      val builder = ScoringResultAvro.newBuilder()
      builder.setPredictionScore(predictionScore)
      builder.setModelId(modelId.getOrElse(DEFAULT_MODEL_ID))
      ids.get(columnsNames(InputColumnsNames.UID)).foreach(builder.setUid(_))
      labelOpt.foreach(builder.setLabel(_))
      weightOpt.foreach(builder.setWeight(_))
      builder.setMetadataMap(metaDataMap)
      builder.build()
    }

    AvroUtils.saveAsAvro(scoringResultAvros, outputDir, ScoringResultAvro.getClassSchema.toString)
  }
}
