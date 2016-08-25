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

  @Test
  def testMakeRandomEffectIdMapWithIdField(): Unit = {
    val record = new GenericData.Record(SchemaBuilder
      .record("testRecordForRandomIdFetch1")
      .namespace("com.linkedin.photon.ml.avro.data")
      .fields()
      .name("userId").`type`().stringType().noDefault()
      .name("jobId").`type`().longType().noDefault()
      .endRecord())

    record.put("userId", "11A")
    record.put("jobId", 112L)
    val map1 = DataProcessingUtils.makeRandomEffectIdMap(record, Set[String]("userId"))
    assertEquals(map1.size, 1)
    assertEquals(map1("userId"), "11A")

    val map2 = DataProcessingUtils.makeRandomEffectIdMap(record, Set[String]("userId", "jobId"))
    assertEquals(map2.size, 2)
    assertEquals(map2("userId"), "11A")
    assertEquals(map2("jobId"), "112")
  }

  @Test
  def testMakeRandomEffectIdMapWithMetadataMap(): Unit = {
    val record = new GenericData.Record(SchemaBuilder
      .record("testRecordForRandomIdFetch2")
      .namespace("com.linkedin.photon.ml.avro.data")
      .fields()
      .name(AvroFieldNames.META_DATA_MAP).`type`().map().values().stringType().noDefault()
      .endRecord())

    val map = new java.util.HashMap[String, String]()
    map.put("userId", "11A")
    map.put("jobId", "112")
    record.put(AvroFieldNames.META_DATA_MAP, map)

    val res = DataProcessingUtils.makeRandomEffectIdMap(record, Set[String]("userId"))
    assertEquals(res.size, 1)
    assertEquals(res("userId"), "11A")

    val res2 = DataProcessingUtils.makeRandomEffectIdMap(record, Set[String]("userId", "jobId"))
    assertEquals(res2.size, 2)
    assertEquals(res2("userId"), "11A")
    assertEquals(res2("jobId"), "112")
  }

  @Test
  def testMakeRandomEffectIdMapWithBothFields(): Unit = {
    // Expecting 1st layer fields override the metadataMap fields

    val record = new GenericData.Record(SchemaBuilder
      .record("testRecordForRandomIdFetch3")
      .namespace("com.linkedin.photon.ml.avro.data")
      .fields()
      .name(AvroFieldNames.META_DATA_MAP).`type`().map().values().stringType().noDefault()
      .name("userId").`type`().stringType().noDefault()
      .name("jobId").`type`().longType().noDefault()
      .endRecord())

    record.put("userId", "11B")
    record.put("jobId", 113L)

    val map = new java.util.HashMap[String, String]()
    map.put("userId", "11A")
    map.put("jobId", "112")
    record.put(AvroFieldNames.META_DATA_MAP, map)

    val res = DataProcessingUtils.makeRandomEffectIdMap(record, Set[String]("userId"))
    assertEquals(res.size, 1)
    assertEquals(res("userId"), "11B")

    val res2 = DataProcessingUtils.makeRandomEffectIdMap(record, Set[String]("userId", "jobId"))
    assertEquals(res2.size, 2)
    assertEquals(res2("userId"), "11B")
    assertEquals(res2("jobId"), "113")
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

    assertTrue(DataProcessingUtils.makeRandomEffectIdMap(emptyRecord, Set[String]()).isEmpty)
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

    DataProcessingUtils.makeRandomEffectIdMap(emptyRecord, Set[String]("userId"))
  }
}
