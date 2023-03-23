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
package com.linkedin.photon.ml.util

import java.lang.{Boolean => JBoolean, Double => JDouble, Float => JFloat, Integer => JInteger, Long => JLong, String => JString}

import org.apache.avro.Schema
import org.apache.avro.generic.{GenericData, GenericRecord}
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.hadoop.mapred.JobConf
import org.testng.Assert._
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.Constants
import com.linkedin.photon.ml.evaluation.EvaluatorType._
import com.linkedin.photon.ml.evaluation.{MultiAUC, MultiPrecisionAtK}
import com.linkedin.photon.ml.test.TestTemplateWithTmpDir

/**
 * Unit tests for [[Utils]].
 */
class UtilsTest extends TestTemplateWithTmpDir {

  import UtilsTest._

  /**
   * Test directory creation.
   */
  @Test
  def testCreateHDFSDir(): Unit = {
    val conf = new JobConf()
    val fs = FileSystem.get(conf)
    val dir = new Path(getTmpDir, "testCreateHDFSDir")

    Utils.createHDFSDir(dir, conf)
    assertTrue(fs.exists(dir))
    assertTrue(fs.isDirectory(dir))

    // Try to create an already created dir, no change or exception should happen
    Utils.createHDFSDir(dir, conf)
    assertTrue(fs.exists(dir))
    assertTrue(fs.isDirectory(dir))
  }

  /**
   * Test directory deletion.
   */
  @Test
  def testDeleteHDFSDir(): Unit = {
    val conf = new JobConf()
    val fs = FileSystem.get(conf)
    val createdDir = new Path(getTmpDir, "testDeleteHDFSDir")

    // Directory not existing, nothing should happen
    Utils.deleteHDFSDir(createdDir, conf)
    assertFalse(fs.exists(createdDir))

    fs.mkdirs(createdDir)
    assertTrue(fs.exists(createdDir))
    Utils.deleteHDFSDir(createdDir, conf)
    assertFalse(fs.exists(createdDir))
  }

  /**
   * Test that a unique feature key can be correctly constructed from a (name, term) pair.
   */
  @Test
  def testGetFeatureKey(): Unit = {
    assertEquals(Utils.getFeatureKey("foo", "bar"), s"foo${Constants.DELIMITER}bar")
    assertEquals(Utils.getFeatureKey("foo", "bar", "\t"), "foo\tbar")
    assertEquals(Utils.getFeatureKey("foo", "bar", " "), "foo bar")
  }

  /**
   * Test that a unique feature key can be correctly extracted from a [[GenericRecord]].
   */
  @Test
  def testGetFeatureKeyFromRecord(): Unit = {
    val record = new TestRecordBuilder()
      .setStringValue("name_value", "term_value")
      .build()

    assertEquals(Utils.getFeatureKey(record, "stringField", "utf8StringField", "\t"), "name_value\tterm_value")
    assertEquals(Utils.getFeatureKey(record, "stringField", "utf8StringField", " "), "name_value term_value")
  }

  /**
   * Test that a unique feature key can be decomposed into the underlying (name, term) pair.
   */
  @Test
  def testGetFeatureNameAndTermFromKey(): Unit = {

    assertEquals(Utils.getFeatureNameFromKey(Utils.getFeatureKey("foo", "bar")), "foo")
    assertEquals(Utils.getFeatureNameFromKey(Utils.getFeatureKey("foo", "bar", "\t"), "\t"), "foo")
    assertEquals(Utils.getFeatureNameFromKey(Utils.getFeatureKey("foo", "bar", " "), " "), "foo")

    assertEquals(Utils.getFeatureTermFromKey(Utils.getFeatureKey("foo", "bar")), "bar")
    assertEquals(Utils.getFeatureTermFromKey(Utils.getFeatureKey("foo", "bar", "\t"), "\t"), "bar")
    assertEquals(Utils.getFeatureTermFromKey(Utils.getFeatureKey("foo", "bar", " "), " "), "bar")
  }

  /**
   * Test that string fields can be read from a [[GenericRecord]].
   */
  @Test
  def testGetStringAvro(): Unit = {
    val record = new TestRecordBuilder()
      .setStringValue("foo", "bar")
      .setFloatValue(1.1f)
      .setIntValue(-1)
      .setLongValue(3L)
      .setDoubleValue(-4.4d)
      .build()

    assertEquals(Utils.getStringAvro(record, "stringField", isNullOK = true), "foo")
    assertEquals(Utils.getStringAvro(record, "utf8StringField"), "bar")
    assertEquals(Utils.getStringAvro(record, "floatField", isNullOK = true), "1.1")
    assertEquals(Utils.getStringAvro(record, "intField"), "-1")
    assertEquals(Utils.getStringAvro(record, "longField", isNullOK = true), "3")
    assertEquals(Utils.getStringAvro(record, "doubleField"), "-4.4")

    // Nullable okay
    assertEquals(Utils.getStringAvro(EMPTY_RECORD, "stringField", isNullOK = true), "")
  }

  /**
   * Test that boolean fields can be read from a [[GenericRecord]].
   */
  @Test
  def testGetBooleanAvro(): Unit = {
    val record = new TestRecordBuilder()
      .setStringValue("true", "false")
      .setBooleanValue(true)
      .build()

    assertTrue(Utils.getBooleanAvro(record, "booleanField"))
    assertTrue(Utils.getBooleanAvro(record, "stringField"))
    assertFalse(Utils.getBooleanAvro(record, "utf8StringField"))

    val record2 = new TestRecordBuilder()
      .setStringValue("false", "true")
      .setBooleanValue(false)
      .build()

    assertFalse(Utils.getBooleanAvro(record2, "booleanField"))
    assertFalse(Utils.getBooleanAvro(record2, "stringField"))
    assertTrue(Utils.getBooleanAvro(record2, "utf8StringField"))
  }

  /**
   * Test that double fields can be read from a [[GenericRecord]].
   */
  @Test
  def testGetDoubleAvro(): Unit = {
    val record = new TestRecordBuilder()
      .setStringValue("1.4", "2.2")
      .setFloatValue(1.1f)
      .setIntValue(-1)
      .setLongValue(3L)
      .setDoubleValue(-4.4d)
      .build()

    assertEquals(Utils.getDoubleAvro(record, "stringField"), 1.4, EPSILON)
    assertEquals(Utils.getDoubleAvro(record, "utf8StringField"), 2.2, EPSILON)
    assertEquals(Utils.getDoubleAvro(record, "floatField"), 1.1, EPSILON)
    assertEquals(Utils.getDoubleAvro(record, "intField"), -1, EPSILON)
    assertEquals(Utils.getDoubleAvro(record, "longField"), 3, EPSILON)
    assertEquals(Utils.getDoubleAvro(record, "doubleField"), -4.4, EPSILON)
  }

  /**
   * Test that float fields can be read from a [[GenericRecord]].
   */
  @Test
  def testGetFloatAvro(): Unit = {
    val record = new TestRecordBuilder()
      .setStringValue("1.4", "2.2")
      .setFloatValue(1.1f)
      .setIntValue(-1)
      .setLongValue(3L)
      .setDoubleValue(-4.4d)
      .build()

    assertEquals(Utils.getFloatAvro(record, "stringField"), 1.4f, EPSILON)
    assertEquals(Utils.getFloatAvro(record, "utf8StringField"), 2.2f, EPSILON)
    assertEquals(Utils.getFloatAvro(record, "floatField"), 1.1f, EPSILON)
    assertEquals(Utils.getFloatAvro(record, "intField"), -1f, EPSILON)
    assertEquals(Utils.getFloatAvro(record, "longField"), 3f, EPSILON)
    assertEquals(Utils.getFloatAvro(record, "doubleField"), -4.4f, EPSILON)
  }

  /**
   * Test that int fields can be read from a [[GenericRecord]].
   */
  @Test
  def testGetIntAvro(): Unit = {
    val record = new TestRecordBuilder()
      .setStringValue("14", "+22")
      .setFloatValue(1.1f)
      .setIntValue(-1)
      .setLongValue(3L)
      .setDoubleValue(-4.4d)
      .build()

    assertEquals(Utils.getIntAvro(record, "stringField"), 14)
    assertEquals(Utils.getIntAvro(record, "utf8StringField"), 22)
    assertEquals(Utils.getIntAvro(record, "floatField"), 1)
    assertEquals(Utils.getIntAvro(record, "intField"), -1)
    assertEquals(Utils.getIntAvro(record, "longField"), 3)
    assertEquals(Utils.getIntAvro(record, "doubleField"), -4)
  }

  /**
   * Test that long fields can be read from a [[GenericRecord]].
   */
  @Test
  def testGetLongAvro(): Unit = {
    val record = new TestRecordBuilder()
      .setStringValue("14", "+22")
      .setFloatValue(1.1f)
      .setIntValue(-1)
      .setLongValue(3L)
      .setDoubleValue(-4.4d)
      .build()

    assertEquals(Utils.getLongAvro(record, "stringField"), 14L)
    assertEquals(Utils.getLongAvro(record, "utf8StringField"), 22L)
    assertEquals(Utils.getLongAvro(record, "floatField"), 1L)
    assertEquals(Utils.getLongAvro(record, "intField"), -1L)
    assertEquals(Utils.getLongAvro(record, "longField"), 3L)
    assertEquals(Utils.getLongAvro(record, "doubleField"), -4L)
  }

  /**
   * Test that map fields can be read from a [[GenericRecord]].
   */
  @Test
  def testGetMapAvro(): Unit = {
    val map = new java.util.HashMap[String, Long]()
    map.put("aaa", 1L)
    map.put("bbb", -2L)

    val record = new TestRecordBuilder()
      .setLongMap(map)
      .build()

    val readMap = Utils.getMapAvro(record, "longValMap")
    assertEquals(readMap.size, 2)
    assertEquals(readMap("aaa"), 1L)
    assertEquals(readMap("bbb"), -2L)

    val map2 = new java.util.HashMap[String, String]()
    map2.put("aaa", "111")
    map2.put("bbb", "222")

    val record2 = new TestRecordBuilder()
      .setStringMap(map2)
      .build()

    val readMap2 = Utils.getMapAvro(record2, "stringValMap")
    assertEquals(readMap2.size, 2)
    assertEquals(readMap2("aaa"), "111")
    assertEquals(readMap2("bbb"), "222")

    val emptyRecord = new TestRecordBuilder().build()

    val readMap3 = Utils.getMapAvro(emptyRecord, "stringValMap", isNullOK = true)
    assertNull(readMap3)
  }

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testGetNonNullableMapAvro(): Unit = {
    val emptyRecord = new TestRecordBuilder().build()
    Utils.getMapAvro(emptyRecord, "stringValMap")
  }

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testGetStringAvroNullableNotOk(): Unit = {
    Utils.getStringAvro(EMPTY_RECORD, "stringField")
  }

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testGetBooleanAvroNullableNotOk(): Unit = {
    Utils.getBooleanAvro(EMPTY_RECORD, "stringField")
  }

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testGetIntAvroNullableNotOk(): Unit = {
    Utils.getIntAvro(EMPTY_RECORD, "stringField")
  }

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testGetLongAvroNullableNotOk(): Unit = {
    Utils.getLongAvro(EMPTY_RECORD, "stringField")
  }

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testGetDoubleAvroNullableNotOk(): Unit = {
    Utils.getDoubleAvro(EMPTY_RECORD, "stringField")
  }

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testGetFloatAvroNullableNotOk(): Unit = {
    Utils.getFloatAvro(EMPTY_RECORD, "stringField")
  }

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testGetBooleanWithEmptyString(): Unit = {
    Utils.getBooleanAvro(EMPTY_STRING_RECORD, "stringField")
  }

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testGetIntAvroWithEmptyString(): Unit = {
    Utils.getIntAvro(EMPTY_STRING_RECORD, "stringField")
  }

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testGetLongAvroWithEmptyString(): Unit = {
    Utils.getLongAvro(EMPTY_STRING_RECORD, "stringField")
  }

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testGetDoubleAvroWithEmptyString(): Unit = {
    Utils.getDoubleAvro(EMPTY_STRING_RECORD, "stringField")
  }

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testGetFloatAvroWithEmptyString(): Unit = {
    Utils.getFloatAvro(EMPTY_STRING_RECORD, "stringField")
  }

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testGetStringAvroOtherType(): Unit = {
    Utils.getStringAvro(FIXED_TYPE_RECORD, "fixedField", isNullOK = true)
  }

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testGetBooleanAvroOtherType(): Unit = {
    Utils.getBooleanAvro(FIXED_TYPE_RECORD, "fixedField")
  }

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testGetDoubleAvroOtherType(): Unit = {
    Utils.getDoubleAvro(FIXED_TYPE_RECORD, "fixedField")
  }

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testGetFloatAvroOtherType(): Unit = {
    Utils.getFloatAvro(FIXED_TYPE_RECORD, "fixedField")
  }

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testGetIntAvroOtherType(): Unit = {
    Utils.getIntAvro(FIXED_TYPE_RECORD, "fixedField")
  }

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testGetLongAvroOtherType(): Unit = {
    Utils.getLongAvro(FIXED_TYPE_RECORD, "fixedField")
  }

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testGetDoubleAvroNaNString(): Unit = {
    val record = new TestRecordBuilder().setStringValue(Double.NaN.toString).build()
    Utils.getDoubleAvro(record, "stringField")
  }

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testGetDoubleAvroInfinityString(): Unit = {
    val record = new TestRecordBuilder().setStringValue(Double.PositiveInfinity.toString).build()
    Utils.getDoubleAvro(record, "stringField")
  }

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testGetFloatAvroNaNString(): Unit = {
    val record = new TestRecordBuilder().setStringValue(Float.NaN.toString).build()
    Utils.getFloatAvro(record, "stringField")
  }

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testGetFloatAvroInfinityString(): Unit = {
    val record = new TestRecordBuilder().setStringValue(Float.PositiveInfinity.toString).build()
    Utils.getFloatAvro(record, "stringField")
  }

  /**
   * Test that evaluators can be correctly parsed.
   */
  @Test
  def testEvaluatorWithName(): Unit = {
    // Test regular evaluators
    val auc = "aUc"
    assertEquals(AUC, Utils.evaluatorWithName(auc))

    val rmse = "RMsE"
    assertEquals(RMSE, Utils.evaluatorWithName(rmse))

    val logisticLoss1 = "lOGIstiClosS"
    assertEquals(LogisticLoss, Utils.evaluatorWithName(logisticLoss1))
    val logisticLoss2 = "logiSTIC_LoSS"
    assertEquals(LogisticLoss, Utils.evaluatorWithName(logisticLoss2))

    val poissonLoss1 = "PoISSonLoSs"
    assertEquals(PoissonLoss, Utils.evaluatorWithName(poissonLoss1))
    val poissonLoss2 = "pOISson_lOSS"
    assertEquals(PoissonLoss, Utils.evaluatorWithName(poissonLoss2))

    val squareLoss1 = "sQUAREDlosS "
    assertEquals(SquaredLoss, Utils.evaluatorWithName(squareLoss1))
    val squareLoss2 = "SquAREd_LOss"
    assertEquals(SquaredLoss, Utils.evaluatorWithName(squareLoss2))

    // Test sharded evaluators
    val precisionAt10 = " prEcIsiON@10:queryId   "
    assertEquals(MultiPrecisionAtK(10, "queryId"), Utils.evaluatorWithName(precisionAt10))

    val shardedAuc = "   AuC:foobar "
    assertEquals(MultiAUC("foobar"), Utils.evaluatorWithName(shardedAuc))
  }

  @DataProvider
  def generateUnrecognizedEvaluators(): Array[Array[Object]] = {
    Array(
      Array("AreaUnderROCCurve"),
      Array("ROC"),
      Array("MSE"),
      Array("RRMSE"),
      Array("logistic"),
      Array("poisson"),
      Array("SVM"),
      Array("squared"),
      Array("121"),
      Array("null"),
      Array("precision"),
      Array("precision@"),
      Array("precision@k"),
      Array("precision@1k"),
      Array("precision@10"),
      Array("precision@queryId"),
      Array("precision@10queryId"),
      Array("precision@10|queryId"),
      Array("precision@10-queryId"),
      Array("auc@queryId"),
      Array("auc-queryId")
    )
  }

  /**
   * Test that unrecognized evaluators are correctly rejected.
   *
   * @param name An invalid evaluator name
   */
  @Test(dataProvider = "generateUnrecognizedEvaluators", expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testUnrecognizedEvaluatorsWithName(name: String): Unit =
    Utils.evaluatorWithName(name)
}

object UtilsTest {

  private val EPSILON: Double = 1e-7
  private val TEST_SCHEMA_STRING =
    """
      |{
      |  "type": "record",
      |  "doc": "A pseudo avro schema created to test Utils class",
      |  "name": "Foo",
      |  "namespace": "com.linkedin.example",
      |  "fields": [
      |    {"name": "stringField", "type": ["null", "string"]},
      |    {"name": "utf8StringField", "type": ["null", "string"]},
      |    {"name": "doubleField", "type": ["null", "double"]},
      |    {"name": "floatField", "type": ["null", "float"]},
      |    {"name": "intField", "type": ["null", "int"]},
      |    {"name": "longField", "type": ["null", "long"]},
      |    {"name": "booleanField", "type": ["null", "boolean"]},
      |    {"name": "fixedField", "type": ["null", {"name": "subrecord1", "type": "fixed", "size": 16}]},
      |    {"name": "stringValMap", "type": ["null", {"type": "map", "values": "string"}]},
      |    {"name": "longValMap", "type": ["null", {"type": "map", "values": "long"}]}
      |  ]
      |}
    """.stripMargin
  private val TEST_SCHEMA = new Schema.Parser().parse(TEST_SCHEMA_STRING)

  private val EMPTY_RECORD = new TestRecordBuilder().build()
  private val EMPTY_STRING_RECORD = new TestRecordBuilder().setStringValue("").build()
  private val FIXED_TYPE_RECORD = new TestRecordBuilder().setFixedValue(Array[Byte]('a'.toByte, 'b'.toByte)).build()

  /**
   * Record builder for the test record.
   */
  private class TestRecordBuilder {
    private val _record = new GenericData.Record(TEST_SCHEMA)

    def setLongMap(map: java.util.Map[String, Long]): TestRecordBuilder = {
      _record.put("longValMap", map)
      this
    }

    def setStringMap(map: java.util.Map[String, String]): TestRecordBuilder = {
      _record.put("stringValMap", map)
      this
    }

    def setStringValue(stringValue: String, utf8StringValue: String = null): TestRecordBuilder = {
      _record.put("stringField", new JString(stringValue))
      _record.put("utf8StringField", utf8StringValue)

      this
    }

    def setDoubleValue(doubleValue: Double): TestRecordBuilder = {
      _record.put("doubleField", new JDouble(doubleValue))
      this
    }

    def setFloatValue(floatValue: Float): TestRecordBuilder = {
      _record.put("floatField", new JFloat(floatValue))
      this
    }

    def setIntValue(intValue: Int): TestRecordBuilder = {
      _record.put("intField", new JInteger(intValue))
      this
    }

    def setLongValue(longValue: Long): TestRecordBuilder = {
      _record.put("longField", new JLong(longValue))
      this
    }

    def setBooleanValue(booleanValue: Boolean): TestRecordBuilder = {
      _record.put("booleanField", new JBoolean(booleanValue))
      this
    }

    def setFixedValue(bytes: Array[Byte]): TestRecordBuilder = {
      _record.put("fixedField", bytes)
      this
    }

    def build(): GenericRecord = _record
  }
}
