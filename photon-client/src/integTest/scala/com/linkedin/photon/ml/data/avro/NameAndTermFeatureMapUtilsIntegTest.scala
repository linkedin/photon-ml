/*
 * Copyright 2019 LinkedIn Corp. All rights reserved.
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

import org.apache.hadoop.fs.Path
import org.apache.spark.rdd.RDD
import org.testng.Assert._
import org.testng.annotations.Test

import com.linkedin.photon.ml.test.{SparkTestUtils, TestTemplateWithTmpDir}

/**
 * Unit tests for the [[NameAndTermFeatureMapUtils]].
 */
class NameAndTermFeatureMapUtilsIntegTest extends SparkTestUtils with TestTemplateWithTmpDir {

  import NameAndTermFeatureMapUtilsIntegTest._

  /**
   * Test that feature bags of [[NameAndTerm]] objects stored in a [[RDD]] can be written to/read from HDFS.
   */
  @Test
  def testReadAndWrite(): Unit = sparkTest("testReadAndWrite") {

    val nameAndTermFeatureBags = FEATURE_BAGS.mapValues { featureBag =>
      sc.parallelize(featureBag.toSeq)
    }
    val tmpDir = new Path(getTmpDir)
    NameAndTermFeatureMapUtils.saveAsTextFiles(nameAndTermFeatureBags, tmpDir.toString, sc)

    val newNameAndTermFeatureMap = NameAndTermFeatureMapUtils.readNameAndTermFeatureMapFromTextFiles(
      tmpDir,
      FEATURE_BAGS.keySet,
      sc)

    assertEquals(newNameAndTermFeatureMap.mapValues(_.collect.toSet), FEATURE_BAGS)
  }

  /**
   * Test that feature bags can be correctly converted to a [[com.linkedin.photon.ml.index.IndexMap]].
   */
  @Test
  def testGetFeatureNameAndTermToIndexMap(): Unit = sparkTest("testGetFeatureNameAndTermToIndexMap") {

    val nameAndTermFeatureMap = FEATURE_BAGS.mapValues {
      nameTermSet => sc.parallelize(nameTermSet.toSeq)
    }
    val indexMap = NameAndTermFeatureMapUtils.getFeatureNameAndTermToIndexMap(
      nameAndTermFeatureMap,
      FEATURE_BAGS.keySet,
      isAddingIntercept = false,
      sc)
    val nameTermSet = FEATURE_BAGS.values.reduce(_ ++ _)

    assertEquals(indexMap.keySet, nameTermSet)
    assertEquals(indexMap.values.toSet, (0 until nameTermSet.size).toSet)
  }
}

object NameAndTermFeatureMapUtilsIntegTest {

  private val FIXED_EFFECT_FEATURE_BAG = Set(NameAndTerm("f1", "t1"), NameAndTerm("f1", "t2"), NameAndTerm("f2", "t1"))
  private val RANDOM_EFFECT_FEATURE_BAG = Set(NameAndTerm("f1", "t1"), NameAndTerm("f3", "t1"))
  private val FEATURE_BAGS = Map("fixed" -> FIXED_EFFECT_FEATURE_BAG, "random" -> RANDOM_EFFECT_FEATURE_BAG)
}
