package com.linkedin.photon.ml.data.avro

import org.apache.hadoop.fs.{FileSystem, Path}
import org.testng.Assert._
import org.testng.annotations.Test

import com.linkedin.photon.ml.io.FeatureShardConfiguration
import com.linkedin.photon.ml.test.{SparkTestUtils, TestTemplateWithTmpDir}

/**
 * Integeration test for AvroDataWriter
 */
class AvroDataWriterIntegTest extends SparkTestUtils with TestTemplateWithTmpDir {

  import AvroDataWriterIntegTest._

  @Test
  def testWrite(): Unit = sparkTest("testRead") {
    val dr = new AvroDataReader()
    val (df, indexMapLoadersMap) = dr.readMerged(inputPath.toString, featureShardConfigurationsMap, numPartitions)
    val outputDir = new Path(getTmpDir)

    assertTrue(df.columns.contains(featureColumn))
    assertTrue(df.columns.contains(responseColumn))
    assertEquals(df.count, 34810)
    assertTrue(indexMapLoadersMap.contains(featureColumn))

    val indexMapLoader = indexMapLoadersMap(featureColumn)
    val writer = new AvroDataWriter
    writer.write(df, outputDir.toString, indexMapLoader, responseColumn, featureColumn, overwrite = true)

    val fs = FileSystem.get(sc.hadoopConfiguration)
    val files = fs.listStatus(outputDir).filter(_.getPath.getName.startsWith("part"))
    assertEquals(files.length, numPartitions)

    val (writeData, _) = dr.read(outputDir.toString, numPartitions)
    assertTrue(writeData.columns.contains(responseColumn))
    assertTrue(writeData.columns.contains(featureColumn))
    assertEquals(writeData.count(), 34810)
  }
}

object AvroDataWriterIntegTest {
  private val inputDir = getClass.getClassLoader.getResource("GameIntegTest/input").getPath
  private val inputPath = new Path(inputDir, "train")
  private val numPartitions = 4
  private val featureColumn = "features"
  private val responseColumn = "response"
  private val featureShardConfigurationsMap = Map(
    featureColumn -> FeatureShardConfiguration(Set("userFeatures", "songFeatures"), hasIntercept = false))
}
