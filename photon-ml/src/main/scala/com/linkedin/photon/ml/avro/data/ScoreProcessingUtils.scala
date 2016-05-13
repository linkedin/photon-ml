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
package com.linkedin.photon.ml.avro.data

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import com.linkedin.photon.avro.generated.ScoringResultAvro
import com.linkedin.photon.ml.avro.AvroIOUtils
import com.linkedin.photon.ml.cli.game.scoring.ScoredItem


/**
 * Some basic functions to read/write scores computed with GAME model from/to HDFS. The current implementation assumes
 * the scores are stored using Avro format.
 */
object ScoreProcessingUtils {

  /**
   * Load the scored items of type [[ScoredItem]] from the given input directory on HDFS
   * @param inputDir The given input directory
   * @param sparkContext The Spark context
   * @return An [[RDD]] of scored items of type [[ScoredItem]]
   */
  protected[ml] def loadScoredItemsFromHDFS(inputDir: String, sparkContext: SparkContext): RDD[ScoredItem] = {
    val scoreAvros = AvroIOUtils.readFromAvro[ScoringResultAvro](sparkContext, inputDir,
      minNumPartitions = sparkContext.defaultParallelism)
    scoreAvros.map { scoreAvro =>
      val score = scoreAvro.getPredictionScore
      val uid = scoreAvro.getUid.toString
      ScoredItem(uid, score)
    }
  }

  /**
   * Save the scored items of type [[ScoredItem]] to the given output directory on HDFS
   * @param scoredItems An [[RDD]] of scored items of type [[ScoredItem]]
   * @param modelId The model's id that used to compute the scores
   * @param outputDir The given output directory
   */
  protected[ml] def saveScoredItemsToHDFS(scoredItems: RDD[ScoredItem], modelId: String, outputDir: String): Unit = {
    val scoringResultAvros = scoredItems.map { case ScoredItem(uid, predictionScore) =>
      val avroFile = ScoringResultAvro.newBuilder().setUid(uid).setModelId(modelId).setPredictionScore(predictionScore)
      avroFile.build()
    }
    AvroIOUtils.saveAsAvro(scoringResultAvros, outputDir, ScoringResultAvro.getClassSchema.toString)
  }
}
