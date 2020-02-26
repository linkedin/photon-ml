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

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame

import com.linkedin.photon.avro.generated.ScoringResultAvro
import com.linkedin.photon.ml.Types.REType
import com.linkedin.photon.ml.cli.game.scoring.ScoredItem

/**
 * Some basic functions to read/write scores computed with GAME model from/to HDFS. The current implementation assumes
 * the scores are stored using Avro format.
 */
object ScoreProcessingUtils {

  val DEFAULT_MODEL_ID = "N/A"

  /**
   * Load the scored items of type [[ScoredItem]] from the given input directory on HDFS.
   *
   * @param inputDir The given input directory
   * @param sparkContext The Spark context
   * @return An [[RDD]] of model ids of type [[String] and scored items of type [[ScoredItem]]
   */
  protected[ml] def loadScoredItemsFromHDFS(inputDir: String, sparkContext: SparkContext): RDD[(String, ScoredItem)] = {

    val scoreAvros = AvroUtils.readAvroFilesInDir[ScoringResultAvro](
      sparkContext,
      inputDir,
      minNumPartitions = sparkContext.defaultParallelism)

    scoreAvros.map { scoreAvro =>
      val score = scoreAvro.getPredictionScore
      val uid = Option(scoreAvro.getUid).map(_.toString)
      val label = Option(scoreAvro.getLabel).map(_.toDouble)
      val weight = Option(scoreAvro.getWeight).map(_.toDouble)
      val ids = scoreAvro.getMetadataMap.asScala.map { case (k, v) => (k.toString, v.toString) }.toMap
      val idsWithUid = uid match {
        case Some(id) => ids + (ResponsePredictionFieldNames.UID -> id)
        case _ => ids
      }
      val modelId = scoreAvro.getModelId.toString
      (modelId, ScoredItem(score, label, weight, idsWithUid))
    }
  }

  /**
   * Save the scored items of type [[ScoredItem]] to the given output directory on HDFS.
   *
   * @param scoredItems An [[DataFrame]] of scored items [score, label, weight]
   * @param modelId The model's id that used to compute the scores
   * @param outputDir The given output directory
   */
  protected[ml] def saveScoredItemsToHDFS(
    scoredItems: DataFrame,
    reTypes: Iterable[REType],
    outputDir: String,
    modelId: Option[String]): Unit = {

    val scoringResultAvros = scoredItems
      .rdd
      .map { row =>
        val predictionScore = row.getDouble(0)
        val label = row.getDouble(1) // Nullable
        val weight = row.getDouble(2) // Nullable
        val ids = reTypes.map(reType => (reType, row.getAs[String](reType))).toMap

        val metaDataMap = collection.mutable.Map(ids.toMap[CharSequence, CharSequence].toSeq: _*).asJava
        val builder = ScoringResultAvro.newBuilder()
        builder.setPredictionScore(predictionScore)
        builder.setModelId(modelId.getOrElse(DEFAULT_MODEL_ID))
        ids.get(ResponsePredictionFieldNames.UID).foreach(builder.setUid(_))
        Option.apply(label).foreach(builder.setLabel(_))
        Option.apply(weight).foreach(builder.setWeight(_))
        builder.setMetadataMap(metaDataMap)
        builder.build()
      }
    AvroUtils.saveAsAvro(scoringResultAvros, outputDir, ScoringResultAvro.getClassSchema.toString)
  }
}
