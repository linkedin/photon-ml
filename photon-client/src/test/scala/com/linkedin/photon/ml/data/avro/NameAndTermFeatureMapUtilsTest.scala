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

import org.apache.hadoop.fs.Path
import org.apache.spark.rdd.RDD
import org.testng.Assert._
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.test.{SparkTestUtils, TestTemplateWithTmpDir}

/**
 * Unit tests for the [[NameAndTermFeatureMapUtils]].
 */
class NameAndTermFeatureMapUtilsTest extends SparkTestUtils with TestTemplateWithTmpDir {

  @DataProvider
  def keyNameTermMap(): Array[Array[Any]] = {
    val fixedEffect = Set(NameAndTerm("f1", "t1"), NameAndTerm("f1", "t2"), NameAndTerm("f2", "t1"))
    val randomEffect = Set(NameAndTerm("f1", "t1"), NameAndTerm("f3", "t1"))

    Array(
      Array("glmix", Map("fixed" -> fixedEffect, "random" -> randomEffect))
    )
  }

  /**
   * Test HDFS write and read of the [[Map]] of feature section key to [[NameAndTerm]] feature [[RDD]].
   */
  @Test(dataProvider = "keyNameTermMap")
  def testReadAndWrite(dir: String, keyNameTermMap: Map[String, Set[NameAndTerm]]): Unit =
    sparkTest("testReadAndWrite") {

    val tmpDir = new Path(getTmpDir, dir)

    val nameAndTermFeatureMap =
      keyNameTermMap.mapValues{
        nameTermSet => sc.parallelize(nameTermSet.toSeq)
      }

    NameAndTermFeatureMapUtils.saveAsTextFiles(nameAndTermFeatureMap, tmpDir.toString, sc)

    val newNameAndTermFeatureMap =
      NameAndTermFeatureMapUtils.readNameAndTermFeatureMapFromTextFiles(tmpDir, keyNameTermMap.keySet, sc)

    assertEquals(newNameAndTermFeatureMap.mapValues(_.collect.toSet), keyNameTermMap)
  }

  /**
   * Test getFeatureNameAndTermToIndexMap.
   */
  @Test(dataProvider = "keyNameTermMap")
  def testGetFeatureNameAndTermToIndexMap(dir: String, keyNameTermMap: Map[String, Set[NameAndTerm]]): Unit =
    sparkTest("testGetFeatureNameAndTermToIndexMap") {
      val nameAndTermFeatureMap =
        keyNameTermMap.mapValues{
          nameTermSet => sc.parallelize(nameTermSet.toSeq)
        }
      val indexMap = NameAndTermFeatureMapUtils.getFeatureNameAndTermToIndexMap(
        nameAndTermFeatureMap, keyNameTermMap.keySet, false, sc)

      val nameTermSet = keyNameTermMap.values.reduce(_ ++ _)

      assertEquals(indexMap.keySet, nameTermSet)
      assertEquals(indexMap.values.toSet, (0 until nameTermSet.size).toSet)
    }

}
