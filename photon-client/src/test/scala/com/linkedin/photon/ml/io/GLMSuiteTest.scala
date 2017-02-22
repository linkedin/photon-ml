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
package com.linkedin.photon.ml.io

import org.testng.Assert
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.test.CommonTestUtils
import com.linkedin.photon.ml.util.{DefaultIndexMap, Utils}

/**
 * This class tests some basic util methods in GLMSuite.
 *
 * Tests that requires a SparkContext are put in [[GLMSuiteIntegTest]].
 */
class GLMSuiteTest {
  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testWithUnsupportedFieldNameType(): Unit = {
    new GLMSuite(FieldNamesType.NONE, false, None, None)
  }

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testWithUnsupportedFieldNameType2(): Unit = {
    new GLMSuite(FieldNamesType.NONE, true, None, None)
  }

  @DataProvider
  def generateInvalidConstraintStrings(): Array[Array[Object]] = {
    val featureKeyToIdMap = Map[String, Int](
      Utils.getFeatureKey("foo", "")->0,
      Utils.getFeatureKey("foo", "bar")->1,
      Utils.getFeatureKey("foo", "baz")->2,
      Utils.getFeatureKey("foo", "qux")->3,
      Utils.getFeatureKey("foo", "quux")->4)
    Array(
      Array(featureKeyToIdMap, """[{"name": "foo", "lowerBound": 0, "upperBound": 1}]"""),
      Array(featureKeyToIdMap, """[{"term": "bar", "lowerBound": 0, "upperBound": 1}]"""),
      Array(featureKeyToIdMap, """[{"name": "foo", "term": "bar", "lowerBound": 1, "upperBound": 0}]"""),
      Array(featureKeyToIdMap,
        """[
             {"name": "foo", "term": "bar", "lowerBound": 0, "upperBound": 1},
             {"name": "*", "term": "*", "lowerBound": 0, "upperBound": 1}
           ]"""),
      Array(featureKeyToIdMap, """[{"name": "*", "term": "foo", "lowerBound": 0, "upperBound": 1}]"""),
      Array(featureKeyToIdMap,
        """[
             {"name": "foo", "term": "*", "lowerBound": 0, "upperBound": 1},
             {"name": "foo", "term": "bar", "lowerBound": 0, "upperBound": 5}
           ]"""),
      Array(featureKeyToIdMap,
        """[
             {"name": "foo", "term": "bar", "lowerBound": 0, "upperBound": 1},
             {"name": "foo", "term": "bar", "lowerBound": 0, "upperBound": 5}
           ]"""),
      Array(featureKeyToIdMap, """[{"name": "foo", "lowerbound": 0, "upperbound": 1}]""")
    )
  }

  @Test(dataProvider = "generateInvalidConstraintStrings", expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testCreateConstraintFeatureMapForInvalidInputs(featureKeyToIdMap: Map[String, Int], constraintString: String): Option[Map[Int, (Double, Double)]] = {
    val suite: GLMSuite = new GLMSuite(FieldNamesType.RESPONSE_PREDICTION, true, Some(constraintString), None)
    suite.featureKeyToIdMap = new DefaultIndexMap(featureKeyToIdMap)
    suite.createConstraintFeatureMap()
  }

  @DataProvider
  def generateValidConstraintStrings(): Array[Array[Object]] = {
    val featureKeyToIdMap = Map[String, Int](
      Utils.getFeatureKey("foo", "")->0,
      Utils.getFeatureKey("foo", "bar")->1,
      Utils.getFeatureKey("foo", "baz")->2,
      Utils.getFeatureKey("qux", "bar")->3,
      GLMSuite.INTERCEPT_NAME_TERM->4,
      Utils.getFeatureKey("qux", "baz")->5,
      Utils.getFeatureKey("qux", "")->6,
      Utils.getFeatureKey("quxl", "")->7)
    Array(
      Array(featureKeyToIdMap, """[{"name": "foo", "term": "baz", "lowerBound": 0, "upperBound": 1}]""",
        Some(Map[Int, (Double, Double)](2->(0.0, 1.0)))),
      Array(featureKeyToIdMap, """[{"name": "nonexistent", "term": "", "lowerBound": 0, "upperBound": 1}]""",
        None),
      Array(featureKeyToIdMap, """[{"name": "foo", "term": "bar", "lowerBound": 0}]""",
        Some(Map[Int, (Double, Double)](1->(0.0, Double.PositiveInfinity)))),
      Array(featureKeyToIdMap, """[{"name": "foo", "term": "bar", "upperBound": 1}]""",
        Some(Map[Int, (Double, Double)](1->(Double.NegativeInfinity, 1.0)))),
      Array(featureKeyToIdMap, """[{"name": "*", "term": "*", "lowerBound": 0, "upperBound": 1}]""",
        Some(Map[Int, (Double, Double)](0->(0.0, 1.0), 1->(0.0, 1.0), 2->(0.0, 1.0), 3->(0.0, 1.0), 5->(0.0, 1.0), 6->(0.0, 1.0), 7->(0.0, 1.0)))),
      Array(featureKeyToIdMap, """[{"name": "qux", "term": "*", "lowerBound": 0, "upperBound": 1}]""",
        Some(Map[Int, (Double, Double)](3->(0.0, 1.0), 5->(0.0, 1.0), 6->(0.0, 1.0)))),
      Array(featureKeyToIdMap,
        """[
             {"name": "foo", "term": "bar", "lowerBound": 0, "upperBound": 1},
             {"name": "qux", "term": "baz", "lowerBound": 0, "upperBound": 1}
           ]""",
        Some(Map[Int, (Double, Double)](1->(0.0, 1.0), 5->(0.0, 1.0))))
    )
  }

  @Test(dataProvider = "generateValidConstraintStrings")
  def testCreateConstraintFeatureMapForValidInputs(featureKeyToIdMap: Map[String, Int], constraintString: String,
                                                   expectedMap: Option[Map[Int, (Double, Double)]]): Unit = {
    val suite: GLMSuite = new GLMSuite(FieldNamesType.RESPONSE_PREDICTION, true, Some(constraintString), None)
    suite.featureKeyToIdMap = new DefaultIndexMap(featureKeyToIdMap)
    val actualMap = suite.createConstraintFeatureMap()
    Assert.assertTrue(CommonTestUtils.compareConstraintMaps(actualMap, expectedMap))
  }
}
