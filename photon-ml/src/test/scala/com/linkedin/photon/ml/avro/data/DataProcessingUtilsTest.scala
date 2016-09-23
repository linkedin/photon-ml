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

import com.linkedin.photon.ml.avro.AvroFieldNames
import org.apache.avro.SchemaBuilder
import org.apache.avro.generic.GenericData
import org.testng.annotations.Test
import org.testng.Assert._

/**
  * This class tests the DataProcessingUtils.
  *
  * @see [[com.linkedin.photon.ml.avro.data.DataProcessingUtils]]
  */
class DataProcessingUtilsTest {

  import DataProcessingUtilsTest._

  @Test
  def testGetGameDatumFromGenericRecordWithUID(): Unit = {
    val record = new GenericData.Record(SchemaBuilder
        .record("testGetGameDatumFromGenericRecordWithUID")
        .namespace("com.linkedin.photon.ml.avro.data")
        .fields()
        .name(AvroFieldNames.UID).`type`().stringType().noDefault()
        .endRecord())
    val uid = "foo"
    record.put(AvroFieldNames.UID, uid)

    val gameDatum = DataProcessingUtils.getGameDatumFromGenericRecord(
      record = record,
      featureShardIdToFeatureSectionKeysMap = Map(),
      featureShardIdToIndexMap = Map(),
      shardIdToFeatureDimensionMap = Map(),
      idTypeSet = Set(),
      isForModelTraining = false)

    assertEquals(gameDatum.idTypeToValueMap.get(AvroFieldNames.UID), Some(uid))
  }

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testGetGameDatumFromGenericRecordWithNoResponse(): Unit = {
    val record = new GenericData.Record(SchemaBuilder
        .record("testGetGameDatumFromGenericRecordWithNoResponse")
        .namespace("com.linkedin.photon.ml.avro.data")
        .fields()
        .name(AvroFieldNames.RESPONSE).`type`().stringType().noDefault()
        .endRecord())

    DataProcessingUtils.getGameDatumFromGenericRecord(
      record = record,
      featureShardIdToFeatureSectionKeysMap = Map(),
      featureShardIdToIndexMap = Map(),
      shardIdToFeatureDimensionMap = Map(),
      idTypeSet = Set(),
      isForModelTraining = true)
  }

  @Test
  def testMakeRandomEffectIdMapWithIdField(): Unit = {
    val record = new GenericData.Record(SchemaBuilder
      .record("testRecordForRandomIdFetch1")
      .namespace("com.linkedin.photon.ml.avro.data")
      .fields()
      .name(USER_ID_NAME).`type`().stringType().noDefault()
      .name(JOB_ID_NAME).`type`().longType().noDefault()
      .endRecord())

    val userIdStr = "11A"
    val jobIdVal = 112L
    val jobIdValStr = "112"

    record.put(USER_ID_NAME, userIdStr)
    record.put(JOB_ID_NAME, jobIdVal)
    val map1 = DataProcessingUtils.getIdTypeToValueMapFromGenericRecord(record, Set[String](USER_ID_NAME))
    assertEquals(map1.size, 1)
    assertEquals(map1(USER_ID_NAME), userIdStr)

    val map2 = DataProcessingUtils.getIdTypeToValueMapFromGenericRecord(record, Set[String](USER_ID_NAME, JOB_ID_NAME))
    assertEquals(map2.size, 2)
    assertEquals(map2(USER_ID_NAME), userIdStr)
    assertEquals(map2(JOB_ID_NAME), jobIdValStr)
  }

  @Test
  def testMakeRandomEffectIdMapWithMetadataMap(): Unit = {
    val record = new GenericData.Record(SchemaBuilder
      .record("testRecordForRandomIdFetch2")
      .namespace("com.linkedin.photon.ml.avro.data")
      .fields()
      .name(AvroFieldNames.META_DATA_MAP).`type`().map().values().stringType().noDefault()
      .endRecord())

    val userIdStr = "11A"
    val jobIdValStr = "112"

    val map = new java.util.HashMap[String, String]()
    map.put(USER_ID_NAME, userIdStr)
    map.put(JOB_ID_NAME, jobIdValStr)
    record.put(AvroFieldNames.META_DATA_MAP, map)

    val res = DataProcessingUtils.getIdTypeToValueMapFromGenericRecord(record, Set[String](USER_ID_NAME))
    assertEquals(res.size, 1)
    assertEquals(res(USER_ID_NAME), userIdStr)

    val res2 = DataProcessingUtils.getIdTypeToValueMapFromGenericRecord(record, Set[String](USER_ID_NAME, JOB_ID_NAME))
    assertEquals(res2.size, 2)
    assertEquals(res2(USER_ID_NAME), userIdStr)
    assertEquals(res2(JOB_ID_NAME), jobIdValStr)
  }

  @Test
  def testMakeRandomEffectIdMapWithBothFields(): Unit = {
    // Expecting 1st layer fields override the metadataMap fields

    val record = new GenericData.Record(SchemaBuilder
      .record("testRecordForRandomIdFetch3")
      .namespace("com.linkedin.photon.ml.avro.data")
      .fields()
      .name(AvroFieldNames.META_DATA_MAP).`type`().map().values().stringType().noDefault()
      .name(USER_ID_NAME).`type`().stringType().noDefault()
      .name(JOB_ID_NAME).`type`().longType().noDefault()
      .endRecord())

    val userId1Str = "11B"
    val userId2Str = "11A"
    val jobId1Val = 113L
    val jobId1Str = "113"
    val jobId2Str = "112"

    record.put(USER_ID_NAME, userId1Str)
    record.put(JOB_ID_NAME, jobId1Val)

    val map = new java.util.HashMap[String, String]()
    map.put(USER_ID_NAME, userId2Str)
    map.put(JOB_ID_NAME, jobId2Str)
    record.put(AvroFieldNames.META_DATA_MAP, map)

    // Ids in metaDataMap will be ignored in this case
    val res = DataProcessingUtils.getIdTypeToValueMapFromGenericRecord(record, Set[String](USER_ID_NAME))
    assertEquals(res.size, 1)
    assertEquals(res(USER_ID_NAME), userId1Str)

    val res2 = DataProcessingUtils.getIdTypeToValueMapFromGenericRecord(record, Set[String](USER_ID_NAME, JOB_ID_NAME))
    assertEquals(res2.size, 2)
    assertEquals(res2(USER_ID_NAME), userId1Str)
    assertEquals(res2(JOB_ID_NAME), jobId1Str)
  }

  @Test
  def testNoRandomEffectIdAtAll(): Unit = {
    // Excepting the method to still proceed but return an empty map

    val emptyRecord = new GenericData.Record(SchemaBuilder
      .record("testRecordForRandomIdFetch3")
      .namespace("com.linkedin.photon.ml.avro.data")
      .fields()
      .name(AvroFieldNames.META_DATA_MAP).`type`().map().values().stringType().noDefault()
      .name("foo").`type`().stringType().noDefault()
      .endRecord())

    assertTrue(DataProcessingUtils.getIdTypeToValueMapFromGenericRecord(emptyRecord, Set[String]()).isEmpty)
  }

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testMakeRandomEffectIdMapWithMissingField(): Unit = {
    // Expecting errors to be raised since nothing is present

    val emptyRecord = new GenericData.Record(SchemaBuilder
      .record("testRecordForRandomIdFetch3")
      .namespace("com.linkedin.photon.ml.avro.data")
      .fields()
      .name(AvroFieldNames.META_DATA_MAP).`type`().map().values().stringType().noDefault()
      .name("foo").`type`().stringType().noDefault()
      .endRecord())

    DataProcessingUtils.getIdTypeToValueMapFromGenericRecord(emptyRecord, Set[String](USER_ID_NAME))
  }
}

object DataProcessingUtilsTest{
  val JOB_ID_NAME = "jobId"
  val USER_ID_NAME = "userId"
}
