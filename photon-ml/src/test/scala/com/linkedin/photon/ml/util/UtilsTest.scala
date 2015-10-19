package com.linkedin.photon.ml.util

import java.lang.{Boolean => JBoolean, Double => JDouble, Float => JFloat, Integer => JInteger, Long => JLong, String => JString}
import java.util.Random

import breeze.linalg.{DenseVector, SparseVector, Vector}
import com.linkedin.photon.ml.io.GLMSuite
import com.linkedin.photon.ml.test.TestTemplateWithTmpDir
import org.apache.avro.Schema
import org.apache.avro.generic.{GenericData, GenericRecord}
import org.apache.avro.util.Utf8
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.hadoop.mapred.JobConf
import org.testng.Assert._
import org.testng.annotations.Test

/**
 * Test the functions in [[Utils]]
 *
 * @author yizhou
 */
class UtilsTest extends TestTemplateWithTmpDir {

  import UtilsTest._

  @Test
  def testCreateHDFSDir(): Unit = {
    val conf = new JobConf()
    val fs = FileSystem.get(conf)
    val dir = getTmpDir + "/testCreateHDFSDir"
    val dirPath = new Path(dir)
    Utils.createHDFSDir(dir, conf)
    assertTrue(fs.exists(dirPath))
    assertTrue(fs.isDirectory(dirPath))

    // Try to create an already created dir, no change or exception should happen
    Utils.createHDFSDir(dir, conf)
    assertTrue(fs.exists(dirPath))
    assertTrue(fs.isDirectory(dirPath))
  }

  @Test
  def testDeleteHDFSDir(): Unit = {
    val conf = new JobConf()
    val fs = FileSystem.get(conf)
    val createdDir = new Path(getTmpDir + "/testDeleteHDFSDir")

    // Directory not existing, nothing should happen
    Utils.deleteHDFSDir(createdDir.toString(), conf)
    assertFalse(fs.exists(createdDir))

    fs.mkdirs(createdDir)
    assertTrue(fs.exists(createdDir))
    Utils.deleteHDFSDir(createdDir.toString(), conf)
    assertFalse(fs.exists(createdDir))
  }

  @Test
  def testInitializeZerosVectorOfSameType(): Unit = {
    val r: Random = new Random(RANDOM_SEED)

    //when the prototype vector is dense
    val prototypeDenseVector = DenseVector.fill(VECTOR_DIMENSION)(r.nextDouble())
    val initializedDenseVector = Utils.initializeZerosVectorOfSameType(prototypeDenseVector)
    assertEquals(prototypeDenseVector.length, initializedDenseVector.length,
      s"Length of the initialized vector (${initializedDenseVector.length}) " +
        s"is different from the prototype vector (${initializedDenseVector.length}})")
    assertTrue(initializedDenseVector.isInstanceOf[DenseVector[Double]],
      s"The initialized dense vector (${initializedDenseVector.getClass}), " +
        s"is not an instance of the prototype vectors' class (${prototypeDenseVector.getClass})")

    //when the prototype vector is sparse
    val indices = Array.tabulate[Int](VECTOR_DIMENSION)(i => i).filter(_ => r.nextBoolean())
    val values = indices.map(_ => r.nextDouble())
    val prototypeSparseVector = new SparseVector[Double](indices, values, VECTOR_DIMENSION)
    val initializedSparseVector = Utils.initializeZerosVectorOfSameType(prototypeSparseVector)
    assertEquals(prototypeSparseVector.length, initializedSparseVector.length,
      s"Length of the initialized vector (${initializedSparseVector.length}) " +
        s"is different from the prototype vector (${prototypeSparseVector.length}})")
    assertEquals(prototypeSparseVector.activeSize, initializedSparseVector.activeSize,
      s"Active size of the initialized vector (${initializedSparseVector.activeSize}) " +
        s"is different from the prototype vector (${prototypeSparseVector.activeSize}})")
    assertTrue(initializedSparseVector.isInstanceOf[SparseVector[Double]],
      s"The initialized sparse vector (${initializedSparseVector.getClass}) " +
        s"is not an instance of the prototype vectors' class (${prototypeSparseVector.getClass})")
  }

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testInitializeZerosVectorOfSameTypeOfUnsupportedVectorType(): Unit = {
    Utils.initializeZerosVectorOfSameType(new MockVector[Double]())
  }


  @Test
  def testGetFeatureKey(): Unit = {
    assertEquals(Utils.getFeatureKey("foo", "bar"), s"foo${GLMSuite.DELIMITER}bar")
    assertEquals(Utils.getFeatureKey("foo", "bar", "\t"), "foo\tbar")
    assertEquals(Utils.getFeatureKey("foo", "bar", " "), "foo bar")
  }

  @Test
  def testGetFeatureKeyFromRecord(): Unit = {
    val record = new TestRecordBuilder()
      .setStringValue("name_value", "term_value")
      .build()

    assertEquals(Utils.getFeatureKey(record, "stringField", "utf8StringField", "\t"), "name_value\tterm_value")
    assertEquals(Utils.getFeatureKey(record, "stringField", "utf8StringField", " "), "name_value term_value")
  }

  @Test
  def testGetFeatureNameFromKey(): Unit = {
    assertEquals(Utils.getFeatureNameFromKey(Utils.getFeatureKey("foo", "bar")), "foo")
    assertEquals(Utils.getFeatureNameFromKey(Utils.getFeatureKey("foo", "bar", "\t"), "\t"), "foo")
    assertEquals(Utils.getFeatureNameFromKey(Utils.getFeatureKey("foo", "bar", " "), " "), "foo")
  }

  @Test
  def testGetFeatureTermFromKey(): Unit = {
    assertEquals(Utils.getFeatureTermFromKey(Utils.getFeatureKey("foo", "bar")), "bar")
    assertEquals(Utils.getFeatureTermFromKey(Utils.getFeatureKey("foo", "bar", "\t"), "\t"), "bar")
    assertEquals(Utils.getFeatureTermFromKey(Utils.getFeatureKey("foo", "bar", " "), " "), "bar")
  }

  @Test
  def testGetStringAvro(): Unit = {
    val record = new TestRecordBuilder()
      .setStringValue("foo", "bar")
      .setFloatValue(1.1f)
      .setIntValue(-1)
      .setLongValue(3L)
      .setDoubleValue(-4.4d)
      .build()

    assertEquals(Utils.getStringAvro(record, "stringField"), "foo")
    assertEquals(Utils.getStringAvro(record, "utf8StringField", true), "bar")
    assertEquals(Utils.getStringAvro(record, "floatField"), "1.1")
    assertEquals(Utils.getStringAvro(record, "intField", true), "-1")
    assertEquals(Utils.getStringAvro(record, "longField"), "3")
    assertEquals(Utils.getStringAvro(record, "doubleField", true), "-4.4")

    // Nullable okay
    assertEquals(Utils.getStringAvro(EMPTY_RECORD, "stringField", true), "")
  }

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

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testGetStringAvroNullableNotOk(): Unit = {
    Utils.getStringAvro(EMPTY_RECORD, "stringField", false)
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
    Utils.getStringAvro(FIXED_TYPE_RECORD, "fixedField", true)
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
    val record = new TestRecordBuilder().setStringValue(Double.NaN.toString()).build()
    Utils.getDoubleAvro(record, "stringField")
  }

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testGetDoubleAvroInfinityString(): Unit = {
    val record = new TestRecordBuilder().setStringValue(Double.PositiveInfinity.toString()).build()
    Utils.getDoubleAvro(record, "stringField")
  }

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testGetFloatAvroNaNString(): Unit = {
    val record = new TestRecordBuilder().setStringValue(Float.NaN.toString()).build()
    Utils.getFloatAvro(record, "stringField")
  }

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testGetFloatAvroInfinityString(): Unit = {
    val record = new TestRecordBuilder().setStringValue(Float.PositiveInfinity.toString()).build()
    Utils.getFloatAvro(record, "stringField")
  }

  @Test
  def testLog1pExp(): Unit = {
    assertEquals(Utils.log1pExp(-1), 0.31326168751, EPSILON)
    assertEquals(Utils.log1pExp(0), 0.69314718056, EPSILON)
    assertEquals(Utils.log1pExp(1), 1.31326168752, EPSILON)
    assertEquals(Utils.log1pExp(10.5), 10.5000275361, EPSILON)
    assertEquals(Utils.log1pExp(100.5), 100.5, EPSILON)
    assertEquals(Utils.log1pExp(10000), 10000, EPSILON)
  }
}

object UtilsTest {
  private val VECTOR_DIMENSION: Int = 10
  private val RANDOM_SEED: Long = 1234567890L
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
      |    {"name": "fixedField", "type": ["null", {"name": "subrecord1", "type": "fixed", "size": 16}]}
      |  ]
      |}
    """.stripMargin
  private val TEST_SCHEMA = new Schema.Parser().parse(TEST_SCHEMA_STRING)

  private val EMPTY_RECORD = new TestRecordBuilder().build()
  private val EMPTY_STRING_RECORD = new TestRecordBuilder().setStringValue("").build()
  private val FIXED_TYPE_RECORD = new TestRecordBuilder().setFixedValue(Array[Byte]('a'.toByte, 'b'.toByte)).build()

  // This is a Vector that mocks a different implementation of breeze Vector, it does nothing meaningful.
  private class MockVector[V] extends Vector[V] {
    override def length: Int = 0

    override def copy: Vector[V] = null

    override def update(i: Int, v: V): Unit = {}

    override def activeSize: Int = 0

    override def apply(i: Int): V = 0d.asInstanceOf[V]

    override def activeIterator: Iterator[(Int, V)] = null

    override def activeKeysIterator: Iterator[Int] = null

    override def activeValuesIterator: Iterator[V] = null

    override def repr: Vector[V] = null
  }

  private class TestRecordBuilder {
    private val _record = new GenericData.Record(TEST_SCHEMA)

    def setStringValue(stringValue: String, utf8StringValue: String = null): TestRecordBuilder = {
      val utf8Val = if (utf8StringValue == null) new Utf8(stringValue) else new Utf8(utf8StringValue)
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
