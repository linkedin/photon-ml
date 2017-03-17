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
package com.linkedin.photon.ml.data

import org.apache.spark.sql.types._
import org.apache.spark.sql.{Row, SQLContext}
import org.testng.Assert._
import org.testng.annotations.Test

import com.linkedin.photon.ml.test.SparkTestUtils

/**
 * Unit tests for GAMEConverters.
 */
class GameConvertersTest extends SparkTestUtils {
  import GameConvertersTest._

  private val uid = "foo"

  @Test
  def testGetGameDatumFromRowWithUID(): Unit = sparkTest("testGetGameDatumFromRowWithUID") {
    val sqlContext = new SQLContext(sc)
    val schema = StructType(Seq(StructField(GameConverters.FieldNames.UID, StringType)))
    val dataFrame = sqlContext.createDataFrame(sc.parallelize(Seq(Row(uid))), schema)

    val gameDatumWithoutResponse = GameConverters.getGameDatumFromRow(
      row = dataFrame.head,
      featureShards = Set(),
      idTypeSet = Set(),
      isResponseRequired = false)

    assertEquals(gameDatumWithoutResponse.idTypeToValueMap.get(GameConverters.FieldNames.UID), Some(uid))
  }

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testGetGameDatumFromRowWithNoResponse(): Unit =
      sparkTest("testGetGameDatumFromGenericRecordWithNoResponse") {

    val sqlContext = new SQLContext(sc)
    val schema = StructType(Seq(StructField(GameConverters.FieldNames.UID, StringType)))
    val dataFrame = sqlContext.createDataFrame(sc.parallelize(Seq(Row(uid))), schema)

    GameConverters.getGameDatumFromRow(
      row = dataFrame.head,
      featureShards = Set(),
      idTypeSet = Set(),
      isResponseRequired = true)
  }

  @Test
  def testMakeRandomEffectTypeMapWithIdField(): Unit = sparkTest("testMakeRandomEffectTypeMapWithIdField") {
    val userIdStr = "11A"
    val jobIdVal = 112L
    val jobIdValStr = "112"

    val sqlContext = new SQLContext(sc)
    val schema = StructType(Seq(
      StructField(GameConverters.FieldNames.UID, StringType),
      StructField(USER_ID_NAME, StringType),
      StructField(JOB_ID_NAME, LongType)))

    val dataFrame = sqlContext.createDataFrame(sc.parallelize(
      Seq(Row(uid, userIdStr, jobIdVal))
    ), schema)
    val row = dataFrame.head

    val map1 = GameConverters.getIdTypeToValueMapFromRow(row, Set[String](USER_ID_NAME))
    assertEquals(map1.size, 1)
    assertEquals(map1(USER_ID_NAME), userIdStr)

    val map2 = GameConverters.getIdTypeToValueMapFromRow(row, Set[String](USER_ID_NAME, JOB_ID_NAME))
    assertEquals(map2.size, 2)
    assertEquals(map2(USER_ID_NAME), userIdStr)
    assertEquals(map2(JOB_ID_NAME), jobIdValStr)
  }

  @Test
  def testMakeRandomEffectTypeMapWithMetadataMap(): Unit = sparkTest("testMakeRandomEffectTypeMapWithMetadataMap") {
    val userIdStr = "11A"
    val jobIdValStr = "112"

    val sqlContext = new SQLContext(sc)
    val schema = StructType(Seq(
      StructField(GameConverters.FieldNames.UID, StringType),
      StructField(GameConverters.FieldNames.META_DATA_MAP, MapType(StringType, StringType, valueContainsNull = false))))

    val dataFrame = sqlContext.createDataFrame(sc.parallelize(
      Seq(Row(uid, Map(USER_ID_NAME -> userIdStr, JOB_ID_NAME -> jobIdValStr)))
    ), schema)
    val row = dataFrame.head

    val res = GameConverters.getIdTypeToValueMapFromRow(row, Set[String](USER_ID_NAME))
    assertEquals(res.size, 1)
    assertEquals(res(USER_ID_NAME), userIdStr)

    val res2 = GameConverters.getIdTypeToValueMapFromRow(row, Set[String](USER_ID_NAME, JOB_ID_NAME))
    assertEquals(res2.size, 2)
    assertEquals(res2(USER_ID_NAME), userIdStr)
    assertEquals(res2(JOB_ID_NAME), jobIdValStr)
  }

  @Test
  def testMakeRandomEffectTypeMapWithBothFields(): Unit = sparkTest("testMakeRandomEffectTypeMapWithMetadataMap") {
    // Expecting 1st layer fields override the metadataMap fields
    val userId1Str = "11B"
    val userId2Str = "11A"
    val jobId1Val = 113L
    val jobId1Str = "113"
    val jobId2Str = "112"

    val sqlContext = new SQLContext(sc)
    val schema = StructType(Seq(
      StructField(GameConverters.FieldNames.UID, StringType),
      StructField(USER_ID_NAME, StringType),
      StructField(JOB_ID_NAME, LongType),
      StructField(GameConverters.FieldNames.META_DATA_MAP, MapType(StringType, StringType, valueContainsNull = false))))

    val dataFrame = sqlContext.createDataFrame(sc.parallelize(
      Seq(Row(uid, userId1Str, jobId1Val, Map(USER_ID_NAME -> userId2Str, JOB_ID_NAME -> jobId2Str)))
    ), schema)
    val row = dataFrame.head

    // Ids in metaDataMap will be ignored in this case
    val res = GameConverters.getIdTypeToValueMapFromRow(row, Set[String](USER_ID_NAME))
    assertEquals(res.size, 1)
    assertEquals(res(USER_ID_NAME), userId1Str)

    val res2 = GameConverters.getIdTypeToValueMapFromRow(row, Set[String](USER_ID_NAME, JOB_ID_NAME))
    assertEquals(res2.size, 2)
    assertEquals(res2(USER_ID_NAME), userId1Str)
    assertEquals(res2(JOB_ID_NAME), jobId1Str)
  }

  @Test
  def testNoRandomEffectTypeAtAll(): Unit = sparkTest("testNoRandomEffectTypeAtAll") {
    // Excepting the method to still proceed but return an empty map
    val sqlContext = new SQLContext(sc)
    val schema = StructType(Seq(
      StructField(GameConverters.FieldNames.UID, StringType),
      StructField(GameConverters.FieldNames.META_DATA_MAP, MapType(StringType, StringType, valueContainsNull = false))))

    val dataFrame = sqlContext.createDataFrame(sc.parallelize(
      Seq(Row(uid, Map()))
    ), schema)
    val row = dataFrame.head

    assertTrue(GameConverters.getIdTypeToValueMapFromRow(row, Set[String]()).isEmpty)
  }

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testMakeRandomEffectTypeMapWithMissingField(): Unit = sparkTest("testMakeRandomEffectTypeMapWithMissingField") {
    // Expecting errors to be raised since nothing is present
    val sqlContext = new SQLContext(sc)
    val schema = StructType(Seq(
      StructField(GameConverters.FieldNames.UID, StringType),
      StructField(GameConverters.FieldNames.META_DATA_MAP, MapType(StringType, StringType, valueContainsNull = false))))

    val dataFrame = sqlContext.createDataFrame(sc.parallelize(
      Seq(Row(uid, Map()))
    ), schema)
    val row = dataFrame.head

    assertTrue(GameConverters.getIdTypeToValueMapFromRow(row, Set[String]()).isEmpty)

    GameConverters.getIdTypeToValueMapFromRow(row, Set[String](USER_ID_NAME))
  }
}

object GameConvertersTest {
  private val JOB_ID_NAME = "jobId"
  private val USER_ID_NAME = "userId"
}
