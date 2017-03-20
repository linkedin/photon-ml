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

import java.io.File

import org.apache.avro.generic.GenericRecord
import org.apache.commons.io.FileUtils
import org.apache.commons.io.filefilter.{FileFilterUtils, PrefixFileFilter}
import org.testng.Assert.assertEquals
import org.testng.annotations.Test

import com.linkedin.photon.avro.generated.FeatureAvro
import com.linkedin.photon.ml.test.{SparkTestUtils, TestTemplateWithTmpDir}

/**
 * This class tests basic IO utilities.
 */
class AvroUtilsTest extends SparkTestUtils with TestTemplateWithTmpDir {
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
    AvroUtils.saveAsAvro[FeatureAvro](outputRdd, outputDir, schemaString)

    // TODO: Rewrite the filter logic when Photon has better file util supports
    val fileFilter = FileFilterUtils
        .notFileFilter(FileFilterUtils.or(new PrefixFileFilter("."), new PrefixFileFilter("_")))
    val files = FileUtils.listFiles(new File(outputDir), fileFilter, null)
    assertEquals(files.size(), 1)

    // Read as specific record
    val specificRdd = AvroUtils.readAvroFilesInDir[FeatureAvro](sc, outputDir, 1)
    val actualSpecific = specificRdd.map(x => (x.getName.toString, x.getTerm.toString, x.getValue)).collect()
    assertEquals(actualSpecific, dataIn)

    // Read as generic record
    val genericRdd = AvroUtils.readAvroFilesInDir[GenericRecord](sc, outputDir, 1)
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
    AvroUtils.saveAsSingleAvro[FeatureAvro](sc, writeData, outputDir, schemaString, forceOverwrite = true)

    // Read as specific record
    val specificList = AvroUtils.readFromSingleAvro[FeatureAvro](sc, outputDir, schemaString = schemaString)
    val actualSpecific = specificList.map(x => (x.getName.toString, x.getTerm.toString, x.getValue)).toArray
    assertEquals(actualSpecific, dataIn)

    // Read as generic record
    val genericList = AvroUtils.readFromSingleAvro[GenericRecord](sc, outputDir, schemaString = schemaString)
    val actualGeneric = genericList.map(x => (x.get("name").toString, x.get("term").toString, x.get("value"))).toArray
    assertEquals(actualGeneric, dataIn)
  }
}
