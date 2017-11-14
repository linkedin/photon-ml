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

import java.io.File

import scala.io.Source

import org.apache.hadoop.fs.{FileSystem, Path}
import org.joda.time.{DateTime, DateTimeUtils}
import org.testng.Assert._
import org.testng.annotations.{AfterClass, BeforeClass, DataProvider, Test}

import com.linkedin.photon.ml.test.{SparkTestUtils, TestTemplateWithTmpDir}

/**
 * This class tests [[IOUtils]].
 */
class IOUtilsIntegTest extends SparkTestUtils with TestTemplateWithTmpDir {

  private val input = getClass.getClassLoader.getResource("IOUtilsTest/input").getPath
  private val baseDir: Path = new Path(input, "daily")
  private val path1: Path = new Path(baseDir, "2016/01/01")
  private val path2: Path = new Path(baseDir, "2016/02/01")
  private val path3: Path = new Path(baseDir, "2016/03/01")
  private val today = "2016-04-01"

  @BeforeClass
  def setup() {
    DateTimeUtils.setCurrentMillisFixed(DateTime.parse(today).getMillis)
  }

  @AfterClass
  def teardown() {
    DateTimeUtils.setCurrentMillisSystem()
  }

  @DataProvider
  def inputPathDataStringProvider(): Array[Array[Any]] = {
    Array(
      Array(DateRange.fromDateString("20160101-20160401"), Seq(path1, path2, path3)),
      Array(DateRange.fromDateString("20160101-20160301"), Seq(path1, path2, path3)),
      Array(DateRange.fromDateString("20160101-20160201"), Seq(path1, path2)),
      Array(DateRange.fromDateString("20160101-20160102"), Seq(path1)),
      Array(DateRange.fromDateString("20160101-20160101"), Seq(path1)),
      Array(DaysRange.fromDaysString("95-1").toDateRange, Seq(path1, path2, path3)),
      Array(DaysRange.fromDaysString("60-1").toDateRange, Seq(path2, path3)),
      Array(DaysRange.fromDaysString("45-1").toDateRange, Seq(path3)))
  }

  /**
   * Test filtering input paths that are within a given date range.
   *
   * @param dateRange The date range to restrict data to
   * @param expectedPaths The expected files
   */
  @Test(dataProvider = "inputPathDataStringProvider")
  def testGetInputPathsWithinDateRange(
      dateRange: DateRange,
      expectedPaths: Seq[Path]): Unit = sparkTest("testGetInputPathsWithinDateRange") {

    assertEquals(
      IOUtils.getInputPathsWithinDateRange(Set(baseDir), dateRange, sc.hadoopConfiguration, errorOnMissing = false),
      expectedPaths)
  }

  /**
   * Test that an empty set of input paths resulting from date range filtering will throw an error.
   */
  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testGetInputPathsWithinDateRangeEmpty(): Unit = sparkTest("testGetInputPathsWithinDateRangeEmpty") {

    IOUtils.getInputPathsWithinDateRange(
      Set(baseDir),
      DateRange.fromDateString("19551105-19551106"),
      sc.hadoopConfiguration,
      errorOnMissing = true)
  }

  /**
   * Test whether an directory existing can be correctly determined.
   */
  @Test
  def testIsDirExisting(): Unit = sparkTest("testIsDirExisting") {

    val dir = new Path(getTmpDir)
    val hadoopConfiguration = sc.hadoopConfiguration

    Utils.deleteHDFSDir(dir, hadoopConfiguration)
    assertFalse(IOUtils.isDirExisting(dir, hadoopConfiguration))
    Utils.createHDFSDir(dir, hadoopConfiguration)
    assertTrue(IOUtils.isDirExisting(dir, hadoopConfiguration))
  }

  /**
   * Test preparing an output directory to receive files.
   */
  @Test
  def testProcessOutputDir(): Unit = sparkTest("testProcessOutputDir") {

    val hadoopConfiguration = sc.hadoopConfiguration

    // Case 1: When the output directory already exists and deleteOutputDirIfExists is true
    val dir1 = new Path(getTmpDir)
    IOUtils.processOutputDir(dir1, deleteOutputDirIfExists = true, hadoopConfiguration)
    assertFalse(IOUtils.isDirExisting(dir1, hadoopConfiguration))

    // Case 2: When the output directory already exists and deleteOutputDirIfExists is false
    val dir2 = new Path(getTmpDir)
    try {
      IOUtils.processOutputDir(dir2, deleteOutputDirIfExists = false, hadoopConfiguration)
    } catch {
      case e: Exception => assertTrue(e.isInstanceOf[IllegalArgumentException])
    } finally {
      assertTrue(IOUtils.isDirExisting(dir2, hadoopConfiguration))
    }

    // Case 3: When the output directory does not exist and deleteOutputDirIfExists is true
    val dir3 = new Path(getTmpDir)
    Utils.deleteHDFSDir(dir3, hadoopConfiguration)
    IOUtils.processOutputDir(dir3, deleteOutputDirIfExists = true, hadoopConfiguration)
    assertFalse(IOUtils.isDirExisting(dir3, hadoopConfiguration))

    // Case 4: When the output directory does not exist and deleteOutputDirIfExists is false
    val dir4 = new Path(getTmpDir)
    Utils.deleteHDFSDir(dir4, hadoopConfiguration)
    IOUtils.processOutputDir(dir4, deleteOutputDirIfExists = false, hadoopConfiguration)
    assertFalse(IOUtils.isDirExisting(dir4, hadoopConfiguration))
  }

  /**
   * Test writing to an HDFS file once.
   */
  @Test
  def testToHDFSFileOnce(): Unit = sparkTest("testToHDFSFileOnce") {

    val fs = FileSystem.get(sc.hadoopConfiguration)

    val res = IOUtils.toHDFSFile(sc, "/tmp/test4")
    { writer => (1 to 3).foreach { i => writer.println(s"$i ") } }

    assert(res.isSuccess)
    assert(Source.fromFile("/tmp/test4").getLines.mkString == "1 2 3 ")
    assert(!fs.exists(new Path("/tmp/test4.prev")))
    assert(!fs.exists(new Path("/tmp/test4-tmp")))

    new File("/tmp/test4").delete
    new File("/tmp/test4.prev").delete
  }

  /**
   * Test writing to an HDFS file repeatedly.
   */
  @Test
  def testToHDFSFileRepeated(): Unit = sparkTest("testToHDFSFileRepeated") {

    val fs = FileSystem.get(sc.hadoopConfiguration)

    for (n <- 1 to 3) {
      val res = IOUtils.toHDFSFile(sc, "/tmp/test5") { writer => (1 to 3).foreach(i => writer.println(s"${n + i} ")) }
      assert(res.isSuccess)
    }

    assert(Source.fromFile("/tmp/test5").getLines.mkString == "4 5 6 ")
    assert(Source.fromFile("/tmp/test5.prev").getLines.mkString == "3 4 5 ")
    assert(!fs.exists(new Path("/tmp/test5-tmp")))

    new File("/tmp/test5").delete
    new File("/tmp/test5.prev").delete
  }
}
