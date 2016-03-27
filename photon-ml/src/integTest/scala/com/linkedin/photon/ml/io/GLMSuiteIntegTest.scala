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
package com.linkedin.photon.ml.io

import java.io.File

import breeze.linalg.SparseVector
import FieldNamesType.FieldNamesType
import com.linkedin.photon.avro.generated.{FeatureSummarizationResultAvro, TrainingExampleAvro}
import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.stat.BasicStatisticalSummary
import com.linkedin.photon.ml.test.{TestTemplateWithTmpDir, SparkTestUtils}
import com.linkedin.photon.ml.util.{DefaultIndexMap, Utils}
import org.apache.avro.Schema
import org.apache.avro.file.{DataFileReader, DataFileWriter}
import org.apache.avro.generic.{GenericRecordBuilder, GenericDatumWriter, GenericRecord}
import org.apache.avro.specific.SpecificDatumReader
import org.apache.spark.SparkException
import org.apache.spark.rdd.RDD
import org.testng.annotations.{DataProvider, Test}

import scala.collection.JavaConversions._
import scala.collection.mutable.ArrayBuffer
import java.util.{ArrayList => JArrayList}

import org.testng.Assert._

/**
 * This class tests components of GLMSuite that requires integration with real RDD or other runtime environments.
 * Also see [[GLMSuiteTest]].
 *
 * @author yizhou
 * @author nkatariy
 */
class GLMSuiteIntegTest extends SparkTestUtils with TestTemplateWithTmpDir {
  import GLMSuiteIntegTest._

  @Test(expectedExceptions = Array(classOf[SparkException]))
  def testLoadFeatureMapWithIllegalFeatureList(): Unit = sparkTest("testLoadFeatureMapWithIllegalFeatureList") {
    val suite = new GLMSuite(FieldNamesType.RESPONSE_PREDICTION, true, None, None)

    val recordBuilder = new GenericRecordBuilder(BAD_RESPONSE_PREDICTION_SCHEMA)
    val records = Array(
      recordBuilder.set("response", 1d)
          .set("features", 99d) // Add an illegal data type
          .build()
    )

    val path = getTmpDir + "/testLoadFeatureMapWithIllegalFeatureList"
    writeToTempDir(records, path, BAD_RESPONSE_PREDICTION_SCHEMA)

    // The featureMap is empty, this actually tests the feature map building process
    suite.readLabeledPointsFromAvro(sc, path, None, 1).count
  }

  @Test(expectedExceptions = Array(classOf[SparkException]))
  def testReadLabeledPointsWithIllegalFeatureList(): Unit = sparkTest("testReadLabeledPointsWithIllegalFeatureList") {
    val suite = new GLMSuite(FieldNamesType.RESPONSE_PREDICTION, true, None, None)

    val recordBuilder = new GenericRecordBuilder(BAD_RESPONSE_PREDICTION_SCHEMA)
    val records = Array(
      recordBuilder.set("response", 1d)
        .set("features", 99d) // Add an illegal data type
        .build()
    )

    val path = getTmpDir + "/testReadLabeledPointsWithIllegalFeatureList"
    writeToTempDir(records, path, BAD_RESPONSE_PREDICTION_SCHEMA)

    suite.featureKeyToIdMap = new DefaultIndexMap(Map[String, Int](("Making map non-empty" -> 1)))

    // Because the map is non empty, now it tests the actual avro record parse method
    suite.readLabeledPointsFromAvro(sc, path, None, 1).count
  }

  @Test(expectedExceptions = Array(classOf[SparkException]))
  def testReadLabeledPointsWithIllegalFeatureList2(): Unit =
      sparkTest("testReadLabeledPointsWithIllegalFeatureList2") {
    val suite = new GLMSuite(FieldNamesType.RESPONSE_PREDICTION, true, None, None)

    val recordBuilder = new GenericRecordBuilder(BAD_RESPONSE_PREDICTION_SCHEMA2)
    val features = new JArrayList[Double]()
    features.add(0.1d)
    val records = Array(
      recordBuilder.set("response", 1d)
        .set("features", features) // Add an illegal data type
        .build()
    )

    val path = getTmpDir + "/testReadLabeledPointsWithIllegalFeatureList2"
    writeToTempDir(records, path, BAD_RESPONSE_PREDICTION_SCHEMA2)

    suite.featureKeyToIdMap = new DefaultIndexMap(Map[String, Int](("Making map non-empty" -> 1)))

    // Because the map is non empty, now it tests the actual avro record parse method
    suite.readLabeledPointsFromAvro(sc, path, None, 1).count
  }

  @DataProvider
  def dataProviderForTestReadLabelPointsFromAvro(): Array[Array[Any]] = {
    Array(
      Array(FieldNamesType.TRAINING_EXAMPLE, true, new TrainingExampleAvroBuilderFactory(),
          TrainingExampleAvro.getClassSchema(), None),
      Array(FieldNamesType.TRAINING_EXAMPLE, false, new TrainingExampleAvroBuilderFactory(),
          TrainingExampleAvro.getClassSchema(), None),
      Array(FieldNamesType.TRAINING_EXAMPLE, true, new TrainingExampleAvroBuilderFactory(),
        TrainingExampleAvro.getClassSchema(), Some("src/integTest/resources/GLMSuiteIntegTest/selectedFeatures.avro")),
      Array(FieldNamesType.TRAINING_EXAMPLE, false, new TrainingExampleAvroBuilderFactory(),
        TrainingExampleAvro.getClassSchema(), Some("src/integTest/resources/GLMSuiteIntegTest/selectedFeatures.avro")),
      Array(FieldNamesType.RESPONSE_PREDICTION, true, new ResponsePredictionAvroBuilderFactory(),
          ResponsePredictionAvroBuilderFactory.SCHEMA, None),
      Array(FieldNamesType.RESPONSE_PREDICTION, false, new ResponsePredictionAvroBuilderFactory(),
          ResponsePredictionAvroBuilderFactory.SCHEMA, None),
      Array(FieldNamesType.RESPONSE_PREDICTION, true, new ResponsePredictionAvroBuilderFactory(),
        ResponsePredictionAvroBuilderFactory.SCHEMA, Some("src/integTest/resources/GLMSuiteIntegTest/selectedFeatures.avro")),
      Array(FieldNamesType.RESPONSE_PREDICTION, false, new ResponsePredictionAvroBuilderFactory(),
        ResponsePredictionAvroBuilderFactory.SCHEMA, Some("src/integTest/resources/GLMSuiteIntegTest/selectedFeatures.avro"))
    )
  }

  @Test
  def testDataProvider(): Unit = {
    /* Some exceptions occurring in a data provider will simply skip the tests without complaining about any exceptions.
     * This could be really undesired and hard to find out.
     *
     * Call the data provider once to ensure that no exceptions are happening in the method.
     */
    dataProviderForTestReadLabelPointsFromAvro()
  }

  @Test(dataProvider = "dataProviderForTestReadLabelPointsFromAvro")
  def testReadLabelPointsFromAvro(fieldNameType: FieldNamesType, addIntercept: Boolean,
                                  builderFactory: TrainingAvroBuilderFactory, avroSchema: Schema,
                                  selectedFeaturesFile: Option[String]): Unit =
      sparkTest("testReadLabelPointsFromTrainingExampleAvroWithIntercept") {
    val suite = new GLMSuite(fieldNameType, addIntercept, None)
    val delim = GLMSuite.DELIMITER

    val avroPath = getTmpDir + "/testReadLabelPointsFromTrainingExampleAvro"

    val records = new ArrayBuffer[GenericRecord]()

    records += builderFactory.newBuilder()
      .setLabel(1.0d)
      .setFeatures(new FeatureAvroListBuilder()
          .append("f1", "t1", 1d)
          .append("f2", "t2", 2d)
          .build())
      .setOffset(1.1d)  // offsets shouldn't have an effect for now
      .build()

    records += builderFactory.newBuilder()
      .setLabel(0d)
      .setFeatures(new FeatureAvroListBuilder()
          .append("f2", "t1", 2d)
          .append("f3", "t2", 3d)
      .build())
      .setOffset(1.2d)
      .build()

    records += builderFactory.newBuilder()
      .setLabel(1.0d)
      .setFeatures(new FeatureAvroListBuilder()
          .append("f1", "t1", 3d)
          .append("f4", "t2", 4d)
      .build())
      .setWeight(2.0)
      .build()

    writeToTempDir(records, avroPath, avroSchema)

    val points = suite.readLabeledPointsFromAvro(sc, avroPath, selectedFeaturesFile, 3)

    checkFeatureMap(suite, addIntercept, selectedFeaturesFile)
    checkPoints(suite, points, avroPath, addIntercept, selectedFeaturesFile)

    // Test second pass reading (should be reusing the previous feature map)
    val secondAvroPath = getTmpDir + "/testReadLabelPointsFromTrainingExampleAvro2"
    val moreRecords = new ArrayBuffer[GenericRecord]()

    moreRecords += builderFactory.newBuilder()
        .setLabel(0d)
        .setFeatures(new FeatureAvroListBuilder()
          .append("f2", "t1", 12d)
          .append("f3", "t2", 13d)
          .append("name should not appear", "term should not appear", 1.11d)
        .build())
        .setOffset(1.2d)
        .build()

    writeToTempDir(moreRecords, secondAvroPath, avroSchema)

    val morePoints = suite.readLabeledPointsFromAvro(sc, secondAvroPath, selectedFeaturesFile, 1)

    checkFeatureMap(suite, addIntercept, selectedFeaturesFile)

    selectedFeaturesFile match {
      case Some(x: String) => {
        assertEquals(morePoints.partitions.length, 1)
        assertEquals(morePoints.count(), 0)
        assertTrue(morePoints.isEmpty())
      }
      case _ => {
        assertEquals(morePoints.partitions.length, 1)
        assertEquals(morePoints.count(), 1)
        // Check the single label point data
        val singlePoint = morePoints.first()
        assertEquals(singlePoint.label, 0d, EPSILON)
        assertEquals(singlePoint.offset, 1.2d, EPSILON)
        assertEquals(singlePoint.weight, 1d, EPSILON)
        val f2t1Id = suite.featureKeyToIdMap(Utils.getFeatureKey("f2", "t1"))
        val f3t2Id = suite.featureKeyToIdMap(Utils.getFeatureKey("f3", "t2"))
        if (addIntercept) {
          val interceptId = suite.featureKeyToIdMap(GLMSuite.INTERCEPT_NAME_TERM)
          assertEquals(singlePoint.features,
            buildSparseVector(singlePoint.features.length)((interceptId, 1d), (f2t1Id, 12d), (f3t2Id, 13d)))
        } else {
          assertEquals(singlePoint.features,
            buildSparseVector(singlePoint.features.length)((f2t1Id, 12d), (f3t2Id, 13d)))
        }
      }
    }
  }

  private def checkFeatureMap(glmSuite: GLMSuite, addIntercept: Boolean, selectedFeaturesFile: Option[String]) = {
    // Check feature map
    val featureMap = glmSuite.featureKeyToIdMap

    val iId = GLMSuite.INTERCEPT_NAME_TERM
    val f1t1Id = Utils.getFeatureKey("f1", "t1")
    val f2t2Id = Utils.getFeatureKey("f2", "t2")
    val f2t1Id = Utils.getFeatureKey("f2", "t1")
    val f3t2Id = Utils.getFeatureKey("f3", "t2")
    val f4t2Id = Utils.getFeatureKey("f4", "t2")

    if (addIntercept) {
      // selected feature file contains two features, ("f1", "t1") and ("f4", "t2")
      selectedFeaturesFile match {
        case Some(x: String) => {
          assertEquals(featureMap.size, 3)
          assertTrue(featureMap.contains(iId))
          assertTrue(featureMap.contains(f1t1Id))
          assertTrue(featureMap.contains(f4t2Id))
        }
        case _ => {
          assertEquals(featureMap.size, 6)
          assertTrue(featureMap.contains(iId))
          assertTrue(featureMap.contains(f1t1Id))
          assertTrue(featureMap.contains(f2t1Id))
          assertTrue(featureMap.contains(f2t2Id))
          assertTrue(featureMap.contains(f3t2Id))
          assertTrue(featureMap.contains(f4t2Id))
        }
      }
    } else {
      selectedFeaturesFile match {
        case Some(x: String) => {
          assertEquals(featureMap.size, 2)
          assertTrue(featureMap.contains(f1t1Id))
          assertTrue(featureMap.contains(f4t2Id))
        }
        case _ => {
          assertEquals(featureMap.size, 5)
          assertTrue(featureMap.contains(f1t1Id))
          assertTrue(featureMap.contains(f2t1Id))
          assertTrue(featureMap.contains(f2t2Id))
          assertTrue(featureMap.contains(f3t2Id))
          assertTrue(featureMap.contains(f4t2Id))
        }
      }
    }
  }

  private def checkPoints(glmSuite: GLMSuite, points: RDD[LabeledPoint], avroPath: String,
                          addIntercept: Boolean, selectedFeaturesFile: Option[String]) = {
    val featureMap = glmSuite.featureKeyToIdMap
    val f1t1Id = Utils.getFeatureKey("f1", "t1")
    val f2t2Id = Utils.getFeatureKey("f2", "t2")
    val f2t1Id = Utils.getFeatureKey("f2", "t1")
    val f3t2Id = Utils.getFeatureKey("f3", "t2")
    val f4t2Id = Utils.getFeatureKey("f4", "t2")

    selectedFeaturesFile match {
      case Some(x: String) => {
        assertEquals(points.partitions.length, 3)
        assertEquals(points.count(), 2)
        assertEquals(points.filter(point => Set[Double](1.1d, 1.2d, 0d).contains(point.offset)).count(), 2)
      }
      case _ => {
        assertEquals(points.partitions.length, 3)
        assertEquals(points.count(), 3)
        assertEquals(points.filter(point => Set[Double](1.1d, 1.2d, 0d).contains(point.offset)).count(), 3)
      }
    }

    points.foreach { point =>
      point.offset match {
        case 1.1d =>
          assertEquals(point.label, 1d, EPSILON)
          assertEquals(point.weight, 1d, EPSILON)
          selectedFeaturesFile match {
            case Some(x: String) => {
              if (addIntercept) {
                val interceptId = featureMap(GLMSuite.INTERCEPT_NAME_TERM)
                assertEquals(point.features,
                  buildSparseVector(point.features.length)((interceptId, 1d), (featureMap(f1t1Id), 1d)))
              } else {
                assertEquals(point.features,
                  buildSparseVector(point.features.length)((featureMap(f1t1Id), 1d)))
              }
            }
            case _ => {
              if (addIntercept) {
                val interceptId = featureMap(GLMSuite.INTERCEPT_NAME_TERM)
                assertEquals(point.features,
                  buildSparseVector(point.features.length)((interceptId, 1d), (featureMap(f1t1Id), 1d), (featureMap(f2t2Id), 2d)))
              } else {
                assertEquals(point.features,
                  buildSparseVector(point.features.length)((featureMap(f1t1Id), 1d), (featureMap(f2t2Id), 2d)))
              }
            }
          }
        case 1.2d =>
          selectedFeaturesFile match {
            case Some(x: String) => fail("Should not see this instance as it has none of the selected features")
            case _ => {
              assertEquals(point.label, 0d, EPSILON)
              assertEquals(point.weight, 1d, EPSILON)
              if (addIntercept) {
                val interceptId = featureMap(GLMSuite.INTERCEPT_NAME_TERM)
                assertEquals(point.features,
                  buildSparseVector(point.features.length)((interceptId, 1d), (featureMap(f2t1Id), 2d), (featureMap(f3t2Id), 3d)))
              } else {
                assertEquals(point.features,
                  buildSparseVector(point.features.length)((featureMap(f2t1Id), 2d), (featureMap(f3t2Id), 3d)))
              }
            }
          }
        case 0d =>
          assertEquals(point.label, 1d, EPSILON)
          assertEquals(point.weight, 2d, EPSILON)
          // all features in this instance are selected so conditioning on selected features file is not necessary
          if (addIntercept) {
            val interceptId = featureMap(GLMSuite.INTERCEPT_NAME_TERM)
            assertEquals(point.features,
              buildSparseVector(point.features.length)((interceptId, 1d), (featureMap(f1t1Id), 3d), (featureMap(f4t2Id), 4d)))
          } else {
            assertEquals(point.features,
              buildSparseVector(point.features.length)((featureMap(f1t1Id), 3d), (featureMap(f4t2Id), 4d)))
          }
        case _ => throw new RuntimeException(s"Observed an unexpected labeled point: ${point}")
      }
    }
  }

  @Test
  def testWriteBasicStatistics(): Unit = sparkTest("testWriteBasicStatistics")  {
    val dim: Int = 5
    val minVector = buildSparseVector(dim)((0, 1.5d), (1, 0d), (2, 0d), (3, 6.7d), (4, 2.33d))
    val maxVector = buildSparseVector(dim)((0, 10d), (1, 0d), (2, 0d), (3, 7d), (4, 4d))
    val normL1Vector = buildSparseVector(dim)((0, 1d), (1, 0d), (2, 0d), (3, 7d), (4, 4d))
    val normL2Vector = buildSparseVector(dim)((0, 2d), (1, 0d), (2, 0d), (3, 8d), (4, 5d))
    val numNonzeros = buildSparseVector(dim)((0, 6d), (1, 0d), (2, 0d), (3, 3d), (4, 89d))
    val meanVector = buildSparseVector(dim)((0, 1.1d), (3, 2.4d), (4, 3.6d))
    val varVector = buildSparseVector(dim)((0, 1d), (3, 7d), (4, 0.5d))


    val summary = BasicStatisticalSummary(
      mean = meanVector,
      variance = varVector,
      count = 101L,
      numNonzeros = numNonzeros,
      max = maxVector,
      min = minVector,
      normL1 = normL1Vector,
      normL2 = normL2Vector,
      meanAbs = meanVector)

    val suite = new GLMSuite(fieldNamesType = FieldNamesType.TRAINING_EXAMPLE, addIntercept = true,
        constraintString = None, offHeapIndexMapLoader = None)
    suite.featureKeyToIdMap = new DefaultIndexMap(Map(
        ("f0" + GLMSuite.DELIMITER -> 0),
        ("f1" + GLMSuite.DELIMITER + "t1" -> 1),
        ("f2" + GLMSuite.DELIMITER -> 2),
        ("f3" + GLMSuite.DELIMITER + "t3" -> 3),
        ("f4" + GLMSuite.DELIMITER -> 4)))

    val tempOut = getTmpDir + "/summary-output"
    suite.writeBasicStatistics(sc, summary, tempOut)

    val reader = DataFileReader.openReader[FeatureSummarizationResultAvro](new File(tempOut + "/part-00000.avro"),
        new SpecificDatumReader[FeatureSummarizationResultAvro]())
    var count = 0
    while (reader.hasNext()) {
      val record = reader.next()
      val feature = record.getFeatureName() + GLMSuite.DELIMITER + record.getFeatureTerm()
      val featureId = suite.featureKeyToIdMap(feature)
      val metrics = record.getMetrics().map {case (key, value) => (String.valueOf(key), value)}
      var foundMatchedOne = true
      featureId match {
        case 0 =>
          assertEquals(feature, "f0" + GLMSuite.DELIMITER)
          assertEquals(metrics("min"), 1.5d, EPSILON)
          assertEquals(metrics("max"), 10d, EPSILON)
          assertEquals(metrics("normL1"), 1d, EPSILON)
          assertEquals(metrics("normL2"), 2d, EPSILON)
          assertEquals(metrics("numNonzeros"), 6d, EPSILON)
          assertEquals(metrics("mean"), 1.1d, EPSILON)
          assertEquals(metrics("variance"), 1d, EPSILON)
        case 1 =>
          assertEquals(feature, "f1" + GLMSuite.DELIMITER + "t1")
          assertEquals(metrics("min"), 0d, EPSILON)
          assertEquals(metrics("max"), 0d, EPSILON)
          assertEquals(metrics("normL1"), 0d, EPSILON)
          assertEquals(metrics("normL2"), 0d, EPSILON)
          assertEquals(metrics("numNonzeros"), 0d, EPSILON)
          assertEquals(metrics("mean"), 0d, EPSILON)
          assertEquals(metrics("variance"), 0d, EPSILON)
        case 2 =>
          assertEquals(feature, "f2" + GLMSuite.DELIMITER)
          assertEquals(metrics("min"), 0d, EPSILON)
          assertEquals(metrics("max"), 0d, EPSILON)
          assertEquals(metrics("normL1"), 0d, EPSILON)
          assertEquals(metrics("normL2"), 0d, EPSILON)
          assertEquals(metrics("numNonzeros"), 0d, EPSILON)
          assertEquals(metrics("mean"), 0d, EPSILON)
          assertEquals(metrics("variance"), 0d, EPSILON)
        case 3 =>
          assertEquals(feature, "f3" + GLMSuite.DELIMITER + "t3")
          assertEquals(metrics("min"), 6.7d, EPSILON)
          assertEquals(metrics("max"), 7d, EPSILON)
          assertEquals(metrics("normL1"), 7d, EPSILON)
          assertEquals(metrics("normL2"), 8d, EPSILON)
          assertEquals(metrics("numNonzeros"), 3d, EPSILON)
          assertEquals(metrics("mean"), 2.4d, EPSILON)
          assertEquals(metrics("variance"), 7d, EPSILON)
        case 4 =>
          assertEquals(feature, "f4" + GLMSuite.DELIMITER)
          assertEquals(metrics("min"), 2.33d, EPSILON)
          assertEquals(metrics("max"), 4d, EPSILON)
          assertEquals(metrics("normL1"), 4d, EPSILON)
          assertEquals(metrics("normL2"), 5d, EPSILON)
          assertEquals(metrics("numNonzeros"), 89d, EPSILON)
          assertEquals(metrics("mean"), 3.6d, EPSILON)
          assertEquals(metrics("variance"), 0.5d, EPSILON)
        case other => foundMatchedOne = false
      }
      if (foundMatchedOne) {
        count += 1
      }
    }
    assertEquals(count, 5)
  }
}

private object GLMSuiteIntegTest {
  val EPSILON = 1e-6

  // A schema that contains illegal features list field
  val BAD_RESPONSE_PREDICTION_SCHEMA = new Schema.Parser().parse(
      new File("src/integTest/resources/GLMSuiteIntegTest/ResponsePredictionError.avsc"))

  // A schema that contains illegal feature item field in the list
  val BAD_RESPONSE_PREDICTION_SCHEMA2 = new Schema.Parser().parse(
    new File("src/integTest/resources/GLMSuiteIntegTest/ResponsePredictionError2.avsc"))

  /**
   * This is a helper methods that builds a sparse vector given an array of (index, value) tuples
   *
   * @param size The size of the vector
   * @param values an array of (index, value) tuples
   * @return a SparseVector instance
   */
  def buildSparseVector(size: Int)(values: (Int, Double)*): SparseVector[Double] = {
    val sortedValues = values.sortBy(x => x._1)
    new SparseVector[Double](sortedValues.map(x => x._1).toArray, sortedValues.map(x => x._2).toArray, size)
  }

  /**
   * A helper method writes a few generic records into an output directory
   *
   * @param it An iterable of generic records
   * @param outputDir The output directory (a part-00000.avro file will be created under that directory)
   * @param schema The avro schema to save as
   */
  def writeToTempDir(it: Iterable[GenericRecord], outputDir: String, schema: Schema): Unit = {
    new File(outputDir).mkdirs()

    val writer = new DataFileWriter[Object](new GenericDatumWriter[Object]())
    try {
      writer.create(schema, new File(outputDir + "/part-00000.avro"))
      it.foreach { record => writer.append(record) }
    } finally {
      writer.close()
    }
  }
}
