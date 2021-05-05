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

import org.apache.avro.Schema.Type._
import org.apache.avro.generic.GenericData
import org.apache.avro.{Schema, SchemaBuilder}
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.sql.types.DataTypes._
import org.apache.spark.sql.types.{DataType, MapType}
import org.testng.Assert._
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.index.DefaultIndexMap
import com.linkedin.photon.ml.util._

class AvroDataReaderTest {
  import AvroDataReaderTest._

  @DataProvider
  def fieldSchemaProvider(): Array[Array[Object]] = {
    Array(
      Array(avroSchema.getFields.get(0).schema, IntegerType),
      Array(avroSchema.getFields.get(1).schema, StringType),
      Array(avroSchema.getFields.get(2).schema, BooleanType),
      Array(avroSchema.getFields.get(3).schema, DoubleType),
      Array(avroSchema.getFields.get(4).schema, FloatType),
      Array(avroSchema.getFields.get(5).schema, LongType),
      Array(avroSchema.getFields.get(6).schema, MapType(StringType, StringType, valueContainsNull = false)),
      Array(avroSchema.getFields.get(7).schema, StringType),
      Array(avroSchema.getFields.get(8).schema, IntegerType),
      Array(avroSchema.getFields.get(9).schema, LongType),
      Array(avroSchema.getFields.get(10).schema, DoubleType),
      Array(avroSchema.getFields.get(11).schema, DoubleType),
      Array(avroSchema.getFields.get(12).schema, IntegerType),
      Array(avroSchema.getFields.get(13).schema, BooleanType),
      Array(avroSchema.getFields.get(14).schema, DoubleType),
      Array(avroSchema.getFields.get(15).schema, FloatType),
      Array(avroSchema.getFields.get(16).schema, LongType))
  }

  @Test(dataProvider = "fieldSchemaProvider")
  def testAvroTypeToSql(avroSchema: Schema, sqlDataType: DataType): Unit = {
    val field = AvroDataReader.avroTypeToSql("testField", avroSchema).get
    assertEquals(field.dataType, sqlDataType)
  }

  @Test
  def testReadColumnValuesFromRecord(): Unit = {
    val fields = Seq(
      AvroDataReader.avroTypeToSql(IntField, avroSchema.getFields.get(0).schema),
      AvroDataReader.avroTypeToSql(StringField, avroSchema.getFields.get(1).schema),
      AvroDataReader.avroTypeToSql(BooleanField, avroSchema.getFields.get(2).schema),
      AvroDataReader.avroTypeToSql(DoubleField, avroSchema.getFields.get(3).schema),
      AvroDataReader.avroTypeToSql(FloatField, avroSchema.getFields.get(4).schema),
      AvroDataReader.avroTypeToSql(LongField, avroSchema.getFields.get(5).schema),
      AvroDataReader.avroTypeToSql(MapField, avroSchema.getFields.get(6).schema),
      AvroDataReader.avroTypeToSql(UnionFieldIntString, avroSchema.getFields.get(7).schema),
      AvroDataReader.avroTypeToSql(UnionFieldIntBoolean, avroSchema.getFields.get(8).schema),
      AvroDataReader.avroTypeToSql(UnionFieldIntLong, avroSchema.getFields.get(9).schema),
      AvroDataReader.avroTypeToSql(UnionFieldFloatDouble, avroSchema.getFields.get(10).schema),
      AvroDataReader.avroTypeToSql(UnionFieldIntLongFloatDouble, avroSchema.getFields.get(11).schema),
      AvroDataReader.avroTypeToSql(NullableIntField, avroSchema.getFields.get(12).schema),
      AvroDataReader.avroTypeToSql(NullableBooleanField, avroSchema.getFields.get(13).schema),
      AvroDataReader.avroTypeToSql(NullableDoubleField, avroSchema.getFields.get(14).schema),
      AvroDataReader.avroTypeToSql(NullableFloatField, avroSchema.getFields.get(15).schema),
      AvroDataReader.avroTypeToSql(NullableLongField, avroSchema.getFields.get(16).schema)).flatten

    val vals = AvroDataReader.readColumnValuesFromRecord(record, fields)
    assertEquals(
      vals,
      Seq(
        IntValue,
        StringValue,
        BooleanValue,
        DoubleValue,
        FloatValue,
        LongValue,
        MapValue,
        UnionIntStringValue.toString,
        UnionIntBooleanValue,
        UnionIntLongValue,
        UnionFloatDoubleValue,
        UnionIntLongFloatDoubleValue,
        null,
        null,
        null,
        null,
        null))
  }

  @Test
  def testReadFeaturesFromRecord(): Unit = {
    val vals = Array(
      (FeatureKey1, FeatureVal1),
      (FeatureKey2, FeatureVal2))

    assertEquals(AvroDataReader.readFeaturesFromRecord(record, Set(FeaturesField)), vals)
  }

  @Test
  def testReadFeatureVectorFromRecord(): Unit = {
    val vector = new SparseVector(2, Array(0, 1), Array(FeatureVal1, FeatureVal2))
    val indexMap = DefaultIndexMap(Seq(FeatureKey1, FeatureKey2))

    assertEquals(AvroDataReader.readFeatureVectorFromRecord(record, Set(FeaturesField), indexMap), vector)
  }

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testReadFeatureVectorFromRecordDuplicateFeatures(): Unit = {
    val record = new GenericData.Record(avroSchema)
    record.put(FeaturesField, List(feature1, feature2, feature1).asJava)

    val indexMap = DefaultIndexMap(Seq(FeatureKey1, FeatureKey2))

    AvroDataReader.readFeatureVectorFromRecord(record, Set(FeaturesField), indexMap)
  }

  @Test
  def testAllNumericTypes(): Unit = {
    assertFalse(AvroDataReader.allNumericTypes(List(INT, STRING)))
    assertTrue(AvroDataReader.allNumericTypes(List(INT, FLOAT)))
    assertFalse(AvroDataReader.allNumericTypes(List(INT, LONG, FLOAT, DOUBLE, STRING)))
    assertTrue(AvroDataReader.allNumericTypes(List(INT, LONG, FLOAT, DOUBLE)))
  }

  @Test
  def testGetDominantNumericType(): Unit = {
    assertEquals(AvroDataReader.getDominantNumericType(List(INT, LONG)), LONG)
    assertEquals(AvroDataReader.getDominantNumericType(List(FLOAT, DOUBLE)), DOUBLE)
    assertEquals(AvroDataReader.getDominantNumericType(List(INT, FLOAT)), FLOAT)
    assertEquals(AvroDataReader.getDominantNumericType(List(INT, LONG, FLOAT, DOUBLE)), DOUBLE)
    assertEquals(AvroDataReader.getDominantNumericType(List(INT, DOUBLE, LONG)), DOUBLE)
  }
}

object AvroDataReaderTest {

  private val IntField = "intField"
  private val IntValue = 7
  private val StringField = "stringField"
  private val StringValue = "ipass"
  private val BooleanField = "booleanField"
  private val BooleanValue = true
  private val DoubleField = "doubleField"
  private val DoubleValue = 13.0D
  private val FloatField = "floatField"
  private val FloatValue = 23.5
  private val LongField = "longField"
  private val LongValue = 31L
  private val MapField = "mapField"
  private val MapValue = Map("a" -> "5")
  private val UnionFieldIntString = "unionFieldIntString"
  private val UnionIntStringValue = 55
  private val UnionFieldIntBoolean = "UnionFieldIntBoolean"
  private val UnionIntBooleanValue = 66
  private val UnionFieldIntLong = "unionFieldIntLong"
  private val UnionIntLongValue = 17
  private val UnionFieldFloatDouble = "unionFieldFloatDouble"
  private val UnionFloatDoubleValue = 43.5
  private val UnionFieldIntLongFloatDouble = "unionFieldIntLongFloatDouble"
  private val UnionIntLongFloatDoubleValue = 5.0D
  private val NullableIntField = "nullableIntField"
  private val NullableBooleanField = "nullableBooleanField"
  private val NullableDoubleField = "nullableDoubleField"
  private val NullableFloatField = "nullableFloatField"
  private val NullableLongField = "nullableLongField"
  private val FeaturesField = "features"
  private val FeatureName1 = "f1"
  private val FeatureVal1 = 1.0
  private val FeatureKey1: String = Utils.getFeatureKey(FeatureName1, "")
  private val FeatureName2 = "f2"
  private val FeatureVal2 = 0.0
  private val FeatureKey2: String = Utils.getFeatureKey(FeatureName2, "")

  private val NameField = "name"
  private val TermField = "term"
  private val ValueField = "value"

  // Use schema identical to NameTermValueAvro but with different namespace, to see if it can be cast to
  // NameTermValueAvro
  private val nameAndTermSchema: Schema = SchemaBuilder
    .record("testNameAndTermSchema")
    .namespace("com.linkedin.photon.ml.avro.data")
    .fields()
    .name(NameField).`type`().stringType().noDefault()
    .name(TermField).`type`().stringType().noDefault()
    .name(ValueField).`type`().doubleType().noDefault()
    .endRecord()

  private val avroSchema: Schema = SchemaBuilder
    .record("testAvroSchema")
    .namespace("com.linkedin.photon.ml.avro.data")
    .fields()
    .name(IntField).`type`().intType().noDefault()
    .name(StringField).`type`().stringType().noDefault()
    .name(BooleanField).`type`().booleanType().noDefault()
    .name(DoubleField).`type`().doubleType().noDefault()
    .name(FloatField).`type`().floatType().noDefault()
    .name(LongField).`type`().longType().noDefault()
    .name(MapField).`type`().map().values().stringType().noDefault()
    .name(UnionFieldIntString).`type`().unionOf().intType().and().stringType().endUnion().noDefault()
    .name(UnionFieldIntBoolean).`type`().unionOf().intType().and().booleanType().endUnion().noDefault()
    .name(UnionFieldIntLong).`type`().unionOf().intType().and().longType().endUnion().noDefault()
    .name(UnionFieldFloatDouble).`type`().unionOf().floatType().and().doubleType().and().nullType().endUnion().noDefault()
    .name(UnionFieldIntLongFloatDouble).`type`().unionOf().intType().and().longType().and().floatType().and().doubleType().and().nullType().endUnion().noDefault()
    .name(NullableIntField).`type`().unionOf().intType().and().nullType().endUnion().noDefault()
    .name(NullableBooleanField).`type`().unionOf().booleanType().and().nullType().endUnion().noDefault()
    .name(NullableDoubleField).`type`().unionOf().doubleType().and().nullType().endUnion().noDefault()
    .name(NullableFloatField).`type`().unionOf().floatType().and().nullType().endUnion().noDefault()
    .name(NullableLongField).`type`().unionOf().longType().and().nullType().endUnion().noDefault()
    .name(FeaturesField).`type`().array().items(nameAndTermSchema).noDefault()
    .endRecord()

  private val record = new GenericData.Record(avroSchema)
  record.put(IntField, IntValue)
  record.put(StringField, StringValue)
  record.put(BooleanField, BooleanValue)
  record.put(DoubleField, DoubleValue)
  record.put(FloatField, FloatValue)
  record.put(LongField, LongValue)
  record.put(MapField, MapValue.asJava)
  record.put(UnionFieldIntString, UnionIntStringValue)
  record.put(UnionFieldIntBoolean, UnionIntBooleanValue)
  record.put(UnionFieldIntLong, UnionIntLongValue)
  record.put(UnionFieldFloatDouble, UnionFloatDoubleValue)
  record.put(UnionFieldIntLongFloatDouble, UnionIntLongFloatDoubleValue)
  record.put(NullableIntField, null)
  record.put(NullableBooleanField, null)
  record.put(NullableDoubleField, null)
  record.put(NullableFloatField, null)
  record.put(NullableLongField, null)

  private val feature1 = new GenericData.Record(nameAndTermSchema)
  feature1.put(NameField, FeatureName1)
  feature1.put(TermField, "")
  feature1.put(ValueField, FeatureVal1)

  private val feature2 = new GenericData.Record(nameAndTermSchema)
  feature2.put(NameField, FeatureName2)
  feature2.put(TermField, "")
  feature2.put(ValueField, FeatureVal2)

  record.put(FeaturesField, List(feature1, feature2).asJava)
}
