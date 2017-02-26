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

import java.io.File
import java.util.{ArrayList => JArrayList}

import scala.collection.JavaConversions._
import scala.collection.mutable.ArrayBuffer

import org.apache.avro.Schema
import org.apache.avro.file.{DataFileReader, DataFileWriter}
import org.apache.avro.generic.{GenericDatumWriter, GenericRecord, GenericRecordBuilder}
import org.apache.avro.specific.SpecificDatumReader
import org.apache.spark.SparkException
import org.apache.spark.rdd.RDD
import org.testng.Assert._
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.avro.generated.{FeatureSummarizationResultAvro, TrainingExampleAvro}
import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.stat.BasicStatisticalSummary
import com.linkedin.photon.ml.test.{SparkTestUtils, TestTemplateWithTmpDir}
import com.linkedin.photon.ml.util.{DefaultIndexMap, IOUtils, Utils}
import com.linkedin.photon.ml.util.VectorUtils.convertIndexAndValuePairArrayToSparseVector

/**
 * This class tests components of GLMSuite that requires integration with real RDD or other runtime environments.
 * Also see [[GLMSuiteTest]].
 */
class GLMSuiteTest extends SparkTestUtils with TestTemplateWithTmpDir {
  import GLMSuiteTest._

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
    suite.readLabeledPointsFromAvro(sc, path, None, 1).count()
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

    suite.featureKeyToIdMap = new DefaultIndexMap(Map[String, Int]("Making map non-empty" -> 1))

    // Because the map is non empty, now it tests the actual avro record parse method
    suite.readLabeledPointsFromAvro(sc, path, None, 1).count()
  }

  @Test(expectedExceptions = Array(classOf[SparkException]))
  def testReadLabeledPointsWithIllegalFeatureList2(): Unit = sparkTest("testReadLabeledPointsWithIllegalFeatureList2") {

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

    suite.featureKeyToIdMap = new DefaultIndexMap(Map[String, Int]("Making map non-empty" -> 1))

    // Because the map is non empty, now it tests the actual avro record parse method
    suite.readLabeledPointsFromAvro(sc, path, None, 1).count()
  }

  @DataProvider
  def dataProviderForTestReadLabelPointsFromAvro(): Array[Array[Any]] = {
    Array(
      Array(FieldNamesType.TRAINING_EXAMPLE, true, new TrainingExampleAvroBuilderFactory(),
        TrainingExampleAvro.getClassSchema, None),
      Array(FieldNamesType.TRAINING_EXAMPLE, false, new TrainingExampleAvroBuilderFactory(),
        TrainingExampleAvro.getClassSchema, None),
      Array(FieldNamesType.TRAINING_EXAMPLE, true, new TrainingExampleAvroBuilderFactory(),
        TrainingExampleAvro.getClassSchema, Some(SELECTED_FEATURES_PATH)),
      Array(FieldNamesType.TRAINING_EXAMPLE, false, new TrainingExampleAvroBuilderFactory(),
        TrainingExampleAvro.getClassSchema, Some(SELECTED_FEATURES_PATH)),
      Array(FieldNamesType.RESPONSE_PREDICTION, true, new ResponsePredictionAvroBuilderFactory(),
          ResponsePredictionAvroBuilderFactory.SCHEMA, None),
      Array(FieldNamesType.RESPONSE_PREDICTION, false, new ResponsePredictionAvroBuilderFactory(),
          ResponsePredictionAvroBuilderFactory.SCHEMA, None),
      Array(FieldNamesType.RESPONSE_PREDICTION, true, new ResponsePredictionAvroBuilderFactory(),
        ResponsePredictionAvroBuilderFactory.SCHEMA, Some(SELECTED_FEATURES_PATH)),
      Array(FieldNamesType.RESPONSE_PREDICTION, false, new ResponsePredictionAvroBuilderFactory(),
        ResponsePredictionAvroBuilderFactory.SCHEMA, Some(SELECTED_FEATURES_PATH))
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
  def testReadLabelPointsFromAvro(
      fieldNameType: FieldNamesType.FieldNamesType,
      addIntercept: Boolean,
      builderFactory: TrainingAvroBuilderFactory, avroSchema: Schema,
      selectedFeaturesFile: Option[String]): Unit = sparkTest("testReadLabelPointsFromAvro") {

    val suite = new GLMSuite(fieldNameType, addIntercept, None)

    val avroPath = getTmpDir

    val records = new ArrayBuffer[GenericRecord]()

    records += builderFactory.newBuilder()
      .setLabel(1.0d)
      .setFeatures(new FeatureAvroListBuilder()
          .append("f1", "t1", 1d)
          .append("f2", "t2", 2d)
          .build())
      .setOffset(1.1d)  // offsets shouldn't have an effect for now
      .build()

    // This data point doesn't have any features belong to the selectedFeatures.avro
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

    // Features that are not contained in the feature map
    val f2t1Id = Utils.getFeatureKey("f2", "t1")
    val f3t2Id = Utils.getFeatureKey("f3", "t2")
    moreRecords += builderFactory.newBuilder()
        .setLabel(0d)
        .setFeatures(new FeatureAvroListBuilder()
          .append("f2", "t1", 12d)
          .append("f3", "t2", 13d)
        .build())
        .setOffset(1.2d)
        .build()

    writeToTempDir(moreRecords, secondAvroPath, avroSchema)

    val morePoints = suite.readLabeledPointsFromAvro(sc, secondAvroPath, selectedFeaturesFile, 1)
    assertEquals(morePoints.partitions.length, 1)
    assertEquals(morePoints.count(), 1)
    // Check the single label point data
    val singlePoint = morePoints.first()
    assertEquals(singlePoint.label, 0d, EPSILON)
    assertEquals(singlePoint.offset, 1.2d, EPSILON)
    assertEquals(singlePoint.weight, 1d, EPSILON)

    checkFeatureMap(suite, addIntercept, selectedFeaturesFile)

    val featureMap = suite.featureKeyToIdMap
    val interceptId = if (addIntercept) {
      featureMap(GLMSuite.INTERCEPT_NAME_TERM)
    } else {
      // Dummy id
      Integer.MAX_VALUE
    }

    val actualIndexAndData = selectedFeaturesFile match {
      case Some(x: String) =>
        // Recall that this data point doesn't contain any features in the selected feature list, thus the feature
        // size is 0
        if (addIntercept) {
          Array((interceptId, 1d))
        } else {
          Array[(Int, Double)]()
        }
      case _ =>
        if (addIntercept) {
          Array((interceptId, 1d), (featureMap(f2t1Id), 12d), (featureMap(f3t2Id), 13d))
        } else {
          Array((featureMap(f2t1Id), 12d), (featureMap(f3t2Id), 13d))
        }
    }
    val numFeatures = singlePoint.features.length
    val actualFeatures = convertIndexAndValuePairArrayToSparseVector(actualIndexAndData, numFeatures)
    assertEquals(singlePoint.features, actualFeatures)
  }

  /**
   *
   * @param glmSuite
   * @param addIntercept
   * @param selectedFeaturesFile
   */
  private def checkFeatureMap(glmSuite: GLMSuite, addIntercept: Boolean, selectedFeaturesFile: Option[String]): Unit = {
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
        case Some(x: String) =>
          assertEquals(featureMap.size, 3)
          assertTrue(featureMap.contains(iId))
          assertTrue(featureMap.contains(f1t1Id))
          assertTrue(featureMap.contains(f4t2Id))
        case _ =>
          assertEquals(featureMap.size, 6)
          assertTrue(featureMap.contains(iId))
          assertTrue(featureMap.contains(f1t1Id))
          assertTrue(featureMap.contains(f2t1Id))
          assertTrue(featureMap.contains(f2t2Id))
          assertTrue(featureMap.contains(f3t2Id))
          assertTrue(featureMap.contains(f4t2Id))
      }
    } else {
      selectedFeaturesFile match {
        case Some(x: String) =>
          assertEquals(featureMap.size, 2)
          assertTrue(featureMap.contains(f1t1Id))
          assertTrue(featureMap.contains(f4t2Id))
        case _ =>
          assertEquals(featureMap.size, 5)
          assertTrue(featureMap.contains(f1t1Id))
          assertTrue(featureMap.contains(f2t1Id))
          assertTrue(featureMap.contains(f2t2Id))
          assertTrue(featureMap.contains(f3t2Id))
          assertTrue(featureMap.contains(f4t2Id))
      }
    }
  }

  /**
   *
   * @param glmSuite
   * @param points
   * @param avroPath
   * @param addIntercept
   * @param selectedFeaturesFile
   */
  private def checkPoints(
      glmSuite: GLMSuite,
      points: RDD[LabeledPoint],
      avroPath: String,
      addIntercept: Boolean,
      selectedFeaturesFile: Option[String]): Unit = {

    val featureMap = glmSuite.featureKeyToIdMap.asInstanceOf[DefaultIndexMap].featureNameToIdMap
    val f1t1Id = Utils.getFeatureKey("f1", "t1")
    val f2t2Id = Utils.getFeatureKey("f2", "t2")
    val f2t1Id = Utils.getFeatureKey("f2", "t1")
    val f3t2Id = Utils.getFeatureKey("f3", "t2")
    val f4t2Id = Utils.getFeatureKey("f4", "t2")

    selectedFeaturesFile match {
      case Some(x: String) =>
        assertEquals(points.partitions.length, 3)
        if (addIntercept) {
          // With intercept added, the number of data points with only intercept-like dummy variable is 1
          assertEquals(points.filter(_.features.activeSize == 1).count(), 1)
        } else {
          // Without intercept, the number of data points with empty feature vector is 1
          assertEquals(points.filter(_.features.activeSize == 0).count(), 1)
        }
        assertEquals(points.count(), 3)
        assertEquals(points.filter(point => Set[Double](1.1d, 1.2d, 0d).contains(point.offset)).count(), 3)
      case _ =>
        assertEquals(points.partitions.length, 3)
        if (addIntercept) {
          // With intercept added, all data points should have 3 features
          assertEquals(points.filter(_.features.activeSize == 3).count(), 3)
        } else {
          // Without intercept, all data points should have 2 features
          assertEquals(points.filter(_.features.activeSize == 2).count(), 3)
        }
        assertEquals(points.count(), 3)
        assertEquals(points.filter(point => Set[Double](1.1d, 1.2d, 0d).contains(point.offset)).count(), 3)
    }

    val interceptId = if (addIntercept) {
      featureMap(GLMSuite.INTERCEPT_NAME_TERM)
    } else {
      // Dummy id
      Integer.MAX_VALUE
    }
    points.foreach { point =>
      val numFeatures = point.features.length
      val actualIndexAndData =
        point.offset match {
          case 1.1d =>
            assertEquals(point.label, 1d, EPSILON)
            assertEquals(point.weight, 1d, EPSILON)
            selectedFeaturesFile match {
              case Some(x: String) =>
                if (addIntercept) {
                  Array((interceptId, 1d), (featureMap(f1t1Id), 1d))
                } else {
                  Array((featureMap(f1t1Id), 1d))
                }
              case _ =>
                if (addIntercept) {
                  Array((interceptId, 1d), (featureMap(f1t1Id), 1d), (featureMap(f2t2Id), 2d))
                } else {
                  Array((featureMap(f1t1Id), 1d), (featureMap(f2t2Id), 2d))
                }
            }
          case 1.2d =>
            assertEquals(point.label, 0d, EPSILON)
            assertEquals(point.weight, 1d, EPSILON)
            selectedFeaturesFile match {
              case Some(x: String) =>
                // Recall that this data point doesn't contain any features in the selected feature list, thus the
                // feature size is 0
              if (addIntercept) {
                  Array((interceptId, 1d))
                } else {
                  Array[(Int, Double)]()
                }
              case _ =>
                if (addIntercept) {
                  Array((interceptId, 1d), (featureMap(f2t1Id), 2d), (featureMap(f3t2Id), 3d))
                } else {
                  Array((featureMap(f2t1Id), 2d), (featureMap(f3t2Id), 3d))
                }
            }
          case 0d =>
            assertEquals(point.label, 1d, EPSILON)
            assertEquals(point.weight, 2d, EPSILON)
            // all features in this instance are selected so conditioning on selected features file is not necessary
            if (addIntercept) {
              Array((interceptId, 1d), (featureMap(f1t1Id), 3d), (featureMap(f4t2Id), 4d))
            } else {
              Array((featureMap(f1t1Id), 3d), (featureMap(f4t2Id), 4d))
            }
          case _ => throw new RuntimeException(s"Observed an unexpected labeled point: $point")
        }
      val actualFeatures = convertIndexAndValuePairArrayToSparseVector(actualIndexAndData, numFeatures)
      assertEquals(point.features, actualFeatures)
    }
  }

  // TODO: this integTest should be relocated to integTests for IOUtils
  @Test
  def testWriteBasicStatistics(): Unit = sparkTest("testWriteBasicStatistics") {
    val dim: Int = 5
    val minVector =
      convertIndexAndValuePairArrayToSparseVector(Array((0, 1.5d), (1, 0d), (2, 0d), (3, 6.7d), (4, 2.33d)), dim)
    val maxVector =
      convertIndexAndValuePairArrayToSparseVector(Array((0, 10d), (1, 0d), (2, 0d), (3, 7d), (4, 4d)), dim)
    val normL1Vector =
      convertIndexAndValuePairArrayToSparseVector(Array((0, 1d), (1, 0d), (2, 0d), (3, 7d), (4, 4d)), dim)
    val normL2Vector =
      convertIndexAndValuePairArrayToSparseVector(Array((0, 2d), (1, 0d), (2, 0d), (3, 8d), (4, 5d)), dim)
    val numNonzeros =
      convertIndexAndValuePairArrayToSparseVector(Array((0, 6d), (1, 0d), (2, 0d), (3, 3d), (4, 89d)), dim)
    val meanVector = convertIndexAndValuePairArrayToSparseVector(Array((0, 1.1d), (3, 2.4d), (4, 3.6d)), dim)
    val varVector = convertIndexAndValuePairArrayToSparseVector(Array((0, 1d), (3, 7d), (4, 0.5d)), dim)


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
        "f0" + GLMSuite.DELIMITER -> 0,
        "f1" + GLMSuite.DELIMITER + "t1" -> 1,
        "f2" + GLMSuite.DELIMITER -> 2,
        "f3" + GLMSuite.DELIMITER + "t3" -> 3,
        "f4" + GLMSuite.DELIMITER -> 4))

    val tempOut = getTmpDir + "/summary-output"
    IOUtils.writeBasicStatistics(sc, summary, tempOut, suite.featureKeyToIdMap)

    val reader = DataFileReader.openReader[FeatureSummarizationResultAvro](new File(tempOut + "/part-00000.avro"),
        new SpecificDatumReader[FeatureSummarizationResultAvro]())
    var count = 0
    while (reader.hasNext) {
      val record = reader.next()
      val feature = record.getFeatureName + GLMSuite.DELIMITER + record.getFeatureTerm
      val featureId = suite.featureKeyToIdMap(feature)
      val metrics = record.getMetrics.map {case (key, value) => (String.valueOf(key), value)}
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

private object GLMSuiteTest {

  private val EPSILON = 1e-6

  private val SELECTED_FEATURES_PATH = "src/integTest/resources/GLMSuiteIntegTest/selectedFeatures.avro"

  // A schema that contains illegal features list field
  private val BAD_RESPONSE_PREDICTION_SCHEMA = new Schema.Parser().parse(
      new File("src/integTest/resources/GLMSuiteIntegTest/ResponsePredictionError.avsc"))

  // A schema that contains illegal feature item field in the list
  private val BAD_RESPONSE_PREDICTION_SCHEMA2 = new Schema.Parser().parse(
    new File("src/integTest/resources/GLMSuiteIntegTest/ResponsePredictionError2.avsc"))

  /**
   * A helper method writes a few generic records into an output directory.
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
