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
package com.linkedin.photon.ml.avro

import org.apache.hadoop.fs.Path
import org.testng.Assert._
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.avro.data.ScoreProcessingUtils
import com.linkedin.photon.ml.cli.game.scoring.ScoredItem
import com.linkedin.photon.ml.test.{SparkTestUtils, TestTemplateWithTmpDir}

class ScoreProcessingUtilsTest extends SparkTestUtils with TestTemplateWithTmpDir {

  @DataProvider
  def scoredItemsProvider():Array[Array[Any]] = {
    val completeScoreItems = Array(
      ScoredItem(predictionScore = 1.0, uid = Some("1"), label = Some(1.0), ids = Map("id1" -> "1", "id2" -> "2")),
      ScoredItem(predictionScore = 0.0, uid = Some("2"), label = Some(0.0), ids = Map("id1" -> "3", "id2" -> "4")),
      ScoredItem(predictionScore = 0.5, uid = Some("3"), label = Some(0.5), ids = Map("id1" -> "5", "id2" -> "6")),
      ScoredItem(predictionScore = -1.0, uid = Some("4"), label = Some(-0.5), ids = Map("id1" -> "7", "id2" -> "8"))
    )
    val scoredItemsWithoutUid = Array(
      ScoredItem(predictionScore = 1.0, uid = None, label = Some(1.0), ids = Map("id1" -> "1", "id2" -> "2")),
      ScoredItem(predictionScore = 0.0, uid = None, label = Some(0.0), ids = Map("id1" -> "3", "id2" -> "4")),
      ScoredItem(predictionScore = 0.5, uid = None, label = Some(0.5), ids = Map("id1" -> "5", "id2" -> "6")),
      ScoredItem(predictionScore = -1.0, uid = None, label = Some(-0.5), ids = Map("id1" -> "7", "id2" -> "8"))
    )
    val scoredItemsWithoutLabel = Array(
      ScoredItem(predictionScore = 1.0, uid = Some("1"), label = None, ids = Map("id1" -> "1", "id2" -> "2")),
      ScoredItem(predictionScore = 0.0, uid = Some("2"), label = None, ids = Map("id1" -> "3", "id2" -> "4")),
      ScoredItem(predictionScore = 0.5, uid = Some("3"), label = None, ids = Map("id1" -> "5", "id2" -> "6")),
      ScoredItem(predictionScore = -1.0, uid = Some("4"), label = None, ids = Map("id1" -> "7", "id2" -> "8"))
    )
    val scoredItemsWithScoreAndLabel = Array(
      ScoredItem(predictionScore = 1.0, uid = None, label = Some(1.0), ids = Map[String, String]()),
      ScoredItem(predictionScore = 0.0, uid = None, label = Some(0.0), ids = Map[String, String]()),
      ScoredItem(predictionScore = 0.5, uid = None, label = Some(0.5), ids = Map[String, String]()),
      ScoredItem(predictionScore = -1.0, uid = None, label = Some(-0.5), ids = Map[String, String]())
    )
    val scoredItemsWithoutIds = Array(
      ScoredItem(predictionScore = 1.0, uid = Some("1"), label = Some(1.0), ids = Map[String, String]()),
      ScoredItem(predictionScore = 0.0, uid = Some("2"), label = Some(0.0), ids = Map[String, String]()),
      ScoredItem(predictionScore = 0.5, uid = Some("3"), label = Some(0.5), ids = Map[String, String]()),
      ScoredItem(predictionScore = -1.0, uid = Some("4"), label = Some(-0.5), ids = Map[String, String]())
    )
    val scoredItemsWithOnlyScores = Array(
      ScoredItem(predictionScore = 1.0, uid = None, label = None, ids = Map[String, String]()),
      ScoredItem(predictionScore = 0.0, uid = None, label = None, ids = Map[String, String]()),
      ScoredItem(predictionScore = 0.5, uid = None, label = None, ids = Map[String, String]()),
      ScoredItem(predictionScore = -1.0, uid = None, label = None, ids = Map[String, String]())
    )
    Array(
      Array("completeScoreItems", completeScoreItems),
      Array("scoredItemsWithoutUid", scoredItemsWithoutUid),
      Array("scoredItemsWithoutLabel", scoredItemsWithoutLabel),
      Array("scoredItemsWithScoreAndLabel", scoredItemsWithScoreAndLabel),
      Array("scoredItemsWithoutIds", scoredItemsWithoutIds),
      Array("scoredItemsWithOnlyScores", scoredItemsWithOnlyScores)
    )
  }

  @Test(dataProvider = "scoredItemsProvider")
  def testLoadAndSaveScoredItems(modelId: String, scoredItems: Array[ScoredItem])
  : Unit = sparkTest("testLoadAndSaveScoredItems") {

    val scoredItemsAsRDD = sc.parallelize(scoredItems, 1)
    val dir = new Path(getTmpDir, "scores").toString
    ScoreProcessingUtils.saveScoredItemsToHDFS(scoredItemsAsRDD, modelId = modelId, dir)
    val loadedModelIdWithScoredItemAsRDD = ScoreProcessingUtils.loadScoredItemsFromHDFS(dir, sc)
    val loadedModelIds = loadedModelIdWithScoredItemAsRDD.map(_._1)

    // Same model Id
    assertTrue(loadedModelIds.collect().forall(_ == modelId))
    val loadedScoredItemAsRDD = loadedModelIdWithScoredItemAsRDD.map(_._2)
    val loadedScoredItem = loadedScoredItemAsRDD.collect()

    // Same scored items
    assertEquals(loadedScoredItem.deep, scoredItems.deep)
  }
}
