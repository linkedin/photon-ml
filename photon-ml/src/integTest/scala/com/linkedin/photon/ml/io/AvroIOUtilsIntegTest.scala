package com.linkedin.photon.ml.io

import java.io.File

import com.linkedin.mlease.spark.test.SparkTestUtils
import com.linkedin.photon.avro.generated.FeatureAvro
import com.linkedin.photon.ml.test.{TestTemplateWithTmpDir, SparkTestUtils}
import org.apache.avro.generic.GenericRecord
import org.apache.commons.io.FileUtils
import org.apache.commons.io.filefilter.{FileFilterUtils, PrefixFileFilter}
import org.testng.Assert.assertEquals
import org.testng.annotations.Test


/**
 * This class tests basic IO utilities.
 * @author dpeng
 */
class AvroIOUtilsIntegTest extends SparkTestUtils with TestTemplateWithTmpDir {
  @Test
  def testAvroReadWrite(): Unit = sparkTest("testAvroReadWrite") {
    val schemaString = FeatureAvro.getClassSchema.toString
    val outputDir = getTmpDir + "/testAvroReadWrite"
    val dataIn = Array(("name1", "term1", 1d), ("name2", "term2", 10d))
    val rawRdd = sc.parallelize(dataIn, 1)
    val outputRdd = rawRdd.map {
      case (name: String, term: String, value: Double) =>
        val builder = FeatureAvro.newBuilder()
        builder.setName(name).setTerm(term).setValue(value).build()
    }
    AvroIOUtils.saveAsAvro[FeatureAvro](outputRdd, outputDir, schemaString)

    // TODO: Rewrite the filter logic when Photon has better file util supports
    val fileFilter = FileFilterUtils.notFileFilter(FileFilterUtils.or(new PrefixFileFilter("."), new PrefixFileFilter("_")))
    val files = FileUtils.listFiles(new File(outputDir), fileFilter, null)
    assertEquals(files.size(), 1)

    // Read as specific record
    val specificRdd = AvroIOUtils.readFromAvro[FeatureAvro](sc, outputDir, 1)
    val actualSpecific = specificRdd.map(x => (x.getName.toString, x.getTerm.toString, x.getValue)).collect()
    assertEquals(actualSpecific, dataIn)

    // Read as generic record
    val genericRdd = AvroIOUtils.readFromAvro[GenericRecord](sc, outputDir, 1)
    val actualGeneric = genericRdd.map(x => (x.get("name").toString, x.get("term").toString, x.get("value"))).collect()
    assertEquals(actualGeneric, dataIn)

  }

  @Test
  def testSingleAvroReadWrite(): Unit = sparkTest("testSingleAvroReadWrite") {
    val schemaString = FeatureAvro.getClassSchema.toString
    val outputDir = getTmpDir + "/testSingleAvroReadWrite"
    val dataIn = Array(("name1", "term1", 1d), ("name2", "term2", 10d))
    val writeData = dataIn.map {
      case (name: String, term: String, value: Double) =>
        val builder = FeatureAvro.newBuilder()
        builder.setName(name).setTerm(term).setValue(value).build()
    }
    AvroIOUtils.saveAsSingleAvro[FeatureAvro](sc, writeData, outputDir, schemaString, forceOverwrite = true)

    // Read as specific record
    val specificList = AvroIOUtils.readFromSingleAvro[FeatureAvro](sc, outputDir, schemaString = schemaString)
    val actualSpecific = specificList.map(x => (x.getName.toString, x.getTerm.toString, x.getValue)).toArray
    assertEquals(actualSpecific, dataIn)

    // Read as generic record
    val genericList = AvroIOUtils.readFromSingleAvro[GenericRecord](sc, outputDir, schemaString = schemaString)
    val actualGeneric = genericList.map(x => (x.get("name").toString, x.get("term").toString, x.get("value"))).toArray
    assertEquals(actualGeneric, dataIn)
  }
}
