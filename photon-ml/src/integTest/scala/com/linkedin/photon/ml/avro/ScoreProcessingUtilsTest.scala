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
import org.testng.Assert.assertEquals
import org.testng.annotations.Test

import com.linkedin.photon.ml.avro.data.ScoreProcessingUtils
import com.linkedin.photon.ml.cli.game.scoring.ScoredItem
import com.linkedin.photon.ml.test.{SparkTestUtils, TestTemplateWithTmpDir}

class ScoreProcessingUtilsTest extends SparkTestUtils with TestTemplateWithTmpDir {

  @Test
  def testLoadAndSaveScoredItems(): Unit = sparkTest("testLoadAndSaveScoredItems") {

    val scoredItems = sc.parallelize(Seq(ScoredItem("1", 1.0), ScoredItem("2", 2.0), ScoredItem("-1", -1.0)))
    val dir = new Path(getTmpDir, "scores").toString
    ScoreProcessingUtils.saveScoredItemsToHDFS(scoredItems, modelId = "", dir)
    val loadedScoredItems = ScoreProcessingUtils.loadScoredItemsFromHDFS(dir, sc)
    val scoredItemsMap = scoredItems.collect().map(scoredItems => (scoredItems.uid, scoredItems)).toMap
    val loadedScoredItemsMap = loadedScoredItems.collect().map(scoredItems => (scoredItems.uid, scoredItems)).toMap
    assertEquals(scoredItemsMap, loadedScoredItemsMap)
  }
}
