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

import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.testng.Assert._
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.avro.generated.ScoringResultAvro
import com.linkedin.photon.ml.cli.game.scoring.ScoredItem
import com.linkedin.photon.ml.data.InputColumnsNames
import com.linkedin.photon.ml.test.{SparkTestUtils, TestTemplateWithTmpDir}

/**
 * Integration tests for [[ScoreProcessingUtils]].
 */
class ScoreProcessingUtilsIntegTest extends SparkTestUtils with TestTemplateWithTmpDir {

  import ScoreProcessingUtilsIntegTest._

  private val completeScoreItems = Array(
    ScoredItem(
      predictionScore = 1.0,
      label = Some(1.0),
      weight = Some(1.0),
      idTagToValueMap = Map((InputColumnsNames.UID.toString, "1"), ("id2", "2"))),
    ScoredItem(
      predictionScore = 0.0,
      label = Some(0.0),
      weight = Some(2.0),
      idTagToValueMap = Map((InputColumnsNames.UID.toString, "3"), ("id2", "4"))),
    ScoredItem(
      predictionScore = 0.5,
      label = Some(0.5),
      weight = Some(-1.0),
      idTagToValueMap = Map((InputColumnsNames.UID.toString, "5"), ("id2", "6"))),
    ScoredItem(
      predictionScore = -1.0,
      label = Some(-0.5),
      weight = Some(0.0),
      idTagToValueMap = Map((InputColumnsNames.UID.toString, "7"), ("id2", "8")))
  )

  private val scoredItemsWithoutUid = Array(
    ScoredItem(predictionScore = 1.0, label = Some(1.0), weight = Some(1.0), idTagToValueMap = Map("id2" -> "2")),
    ScoredItem(predictionScore = 0.0, label = Some(0.0), weight = Some(2.0), idTagToValueMap = Map("id2" -> "4")),
    ScoredItem(predictionScore = 0.5, label = Some(0.5), weight = Some(-1.0), idTagToValueMap = Map("id2" -> "6")),
    ScoredItem(predictionScore = -1.0, label = Some(-0.5), weight = Some(0.0), idTagToValueMap = Map("id2" -> "8"))
  )

  private val scoredItemsWithoutLabel = Array(
    ScoredItem(
      predictionScore = 1.0,
      label = None,
      weight = Some(1.0),
      idTagToValueMap = Map((InputColumnsNames.UID.toString, "1"), ("id2", "2"))),
    ScoredItem(
      predictionScore = 0.0,
      label = None,
      weight = Some(2.0),
      idTagToValueMap = Map((InputColumnsNames.UID.toString, "3"), ("id2", "4"))),
    ScoredItem(
      predictionScore = 0.5,
      label = None,
      weight = Some(-1.0),
      idTagToValueMap = Map((InputColumnsNames.UID.toString, "5"), ("id2", "6"))),
    ScoredItem(
      predictionScore = -1.0,
      label = None,
      weight = Some(0.0),
      idTagToValueMap = Map((InputColumnsNames.UID.toString, "7"), ("id2", "8")))
  )

  private val scoredItemsWithoutWeight = Array(
    ScoredItem(
      predictionScore = 1.0,
      label = Some(1.0),
      weight = None,
      idTagToValueMap = Map((InputColumnsNames.UID.toString, "1"), ("id2", "2"))),
    ScoredItem(
      predictionScore = 0.0,
      label = Some(0.0),
      weight = None,
      idTagToValueMap = Map((InputColumnsNames.UID.toString, "3"), ("id2", "4"))),
    ScoredItem(
      predictionScore = 0.5,
      label = Some(0.5),
      weight = None,
      idTagToValueMap = Map((InputColumnsNames.UID.toString, "5"), ("id2", "6"))),
    ScoredItem(
      predictionScore = -1.0,
      label = Some(-0.5),
      weight = None,
      idTagToValueMap = Map((InputColumnsNames.UID.toString, "7"), ("id2", "8")))
  )

  private val scoredItemsWithoutIds = Array(
    ScoredItem(predictionScore = 1.0, label = Some(1.0), weight = Some(1.0), idTagToValueMap = Map[String, String]()),
    ScoredItem(predictionScore = 0.0, label = Some(0.0), weight = Some(2.0), idTagToValueMap = Map[String, String]()),
    ScoredItem(predictionScore = 0.5, label = Some(0.5), weight = Some(-1.0), idTagToValueMap = Map[String, String]()),
    ScoredItem(predictionScore = -1.0, label = Some(-0.5), weight = Some(0.0), idTagToValueMap = Map[String, String]())
  )

  private val scoredItemsWithOnlyScores = Array(
    ScoredItem(predictionScore = 1.0, label = None, weight = None, idTagToValueMap = Map[String, String]()),
    ScoredItem(predictionScore = 0.0, label = None, weight = None, idTagToValueMap = Map[String, String]()),
    ScoredItem(predictionScore = 0.5, label = None, weight = None, idTagToValueMap = Map[String, String]()),
    ScoredItem(predictionScore = -1.0, label = None, weight = None, idTagToValueMap = Map[String, String]())
  )

  private val mixedScoreItems = Array(
    ScoredItem(
      predictionScore = 1.0,
      label = None,
      weight = Some(1.0),
      idTagToValueMap = Map((InputColumnsNames.UID.toString, "1"), ("id2", "2"))),
    ScoredItem(
      predictionScore = 0.0,
      label = Some(0.0),
      weight = None,
      idTagToValueMap = Map((InputColumnsNames.UID.toString, "3"), ("id2", "4"))),
    ScoredItem(
      predictionScore = 0.5,
      label = Some(0.5),
      weight = Some(-1.0),
      idTagToValueMap = Map[String, String]()),
    ScoredItem(
      predictionScore = -1.0,
      label = Some(-0.5),
      weight = Some(0.0),
      idTagToValueMap = Map((InputColumnsNames.UID.toString, "7"), ("id2", "8")))
  )

  @DataProvider
  def scoredItemsProvider():Array[Array[Any]] = {
    Array(
      Array("completeScoreItems", completeScoreItems),
      Array("scoredItemsWithoutUid", scoredItemsWithoutUid),
      Array("scoredItemsWithoutLabel", scoredItemsWithoutLabel),
      Array("scoredItemsWithoutWeight", scoredItemsWithoutWeight),
      Array("scoredItemsWithoutIds", scoredItemsWithoutIds),
      Array("scoredItemsWithOnlyScores", scoredItemsWithOnlyScores),
      Array("mixedScoreItems", mixedScoreItems)
    )
  }

  @Test(dataProvider = "scoredItemsProvider")
  def testLoadAndSaveScoredItems(modelId: String, scoredItems: Array[ScoredItem]): Unit =
    sparkTest("testLoadAndSaveScoredItems") {
      val scoredItemsAsRDD = sc.parallelize(scoredItems, 1)
      val dir = new Path(getTmpDir, "scores").toString
      ScoreProcessingUtils.saveScoredItemsToHDFS(scoredItemsAsRDD, dir, Some(modelId), InputColumnsNames())
      val loadedModelIdWithScoredItemAsRDD = loadScoredItemsFromHDFS(dir, sc)
      val loadedModelIds = loadedModelIdWithScoredItemAsRDD.map(_._1)

      // Same model Id
      assertTrue(loadedModelIds.collect().forall(_ == modelId))
      val loadedScoredItemAsRDD = loadedModelIdWithScoredItemAsRDD.map(_._2)
      val loadedScoredItem = loadedScoredItemAsRDD.collect()

      // Same scored items
      assertEquals(loadedScoredItem.deep, scoredItems.deep)

      // Same unique ids
      val loadedUids = loadedScoredItem.map(_.idTagToValueMap.get(InputColumnsNames.UID.toString.toString))
      val uids = scoredItems.map(_.idTagToValueMap.get(InputColumnsNames.UID.toString.toString))
      assertEquals(loadedUids.deep, uids.deep)
    }
}

object ScoreProcessingUtilsIntegTest {

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
        case Some(id) => ids + (InputColumnsNames.UID.toString.toString -> id)
        case _ => ids
      }
      val modelId = scoreAvro.getModelId.toString
      (modelId, ScoredItem(score, label, weight, idsWithUid))
    }
  }
}
