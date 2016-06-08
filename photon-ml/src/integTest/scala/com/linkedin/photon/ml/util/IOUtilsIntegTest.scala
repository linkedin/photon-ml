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
package com.linkedin.photon.ml.util

import com.linkedin.photon.ml.test.{SparkTestUtils, TestTemplateWithTmpDir}
import org.joda.time.DateTime
import org.joda.time.DateTimeUtils
import org.testng.Assert._
import org.testng.annotations.{AfterClass, BeforeClass, DataProvider, Test}

/**
 * This class tests IOUtils
 */
class IOUtilsIntegTest extends SparkTestUtils with TestTemplateWithTmpDir {

  val baseDir = getClass.getClassLoader.getResource("IOUtilsTest/input").getPath
  val path1 = s"$baseDir/daily/2016/01/01"
  val path2 = s"$baseDir/daily/2016/02/01"
  val path3 = s"$baseDir/daily/2016/03/01"
  val today = "2016-04-01"

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
      Array(baseDir.toString, DateRange.fromDates("20160101-20160401"), Seq(path1, path2, path3)),
      Array(baseDir.toString, DateRange.fromDates("20160101-20160301"), Seq(path1, path2, path3)),
      Array(baseDir.toString, DateRange.fromDates("20160101-20160201"), Seq(path1, path2)),
      Array(baseDir.toString, DateRange.fromDates("20160101-20160102"), Seq(path1)),
      Array(baseDir.toString, DateRange.fromDates("20160101-20160101"), Seq(path1)),
      Array(baseDir.toString, DateRange.fromDaysAgo(95, 1), Seq(path1, path2, path3)),
      Array(baseDir.toString, DateRange.fromDaysAgo(60, 1), Seq(path2, path3)),
      Array(baseDir.toString, DateRange.fromDaysAgo(45, 1), Seq(path3)))
  }

  @Test(dataProvider = "inputPathDataStringProvider")
  def testGetInputPathsWithinDateRange(
      dir: String,
      dateRange: DateRange,
      expectedPaths: Seq[String]): Unit = sparkTest("testGetInputPathsWithinDateRange") {

    val paths = IOUtils.getInputPathsWithinDateRange(
      Seq(dir), dateRange, sc.hadoopConfiguration, errorOnMissing = false)
    assertEquals(paths, expectedPaths)
  }

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testGetInputPathsWithinDateRangeEmpty(): Unit = sparkTest("testGetInputPathsWithinDateRangeEmpty") {
    IOUtils.getInputPathsWithinDateRange(
      Seq(baseDir), DateRange.fromDates("19551105-19551106"), sc.hadoopConfiguration, errorOnMissing = true)
  }

  @Test
  def testIsDirExisting(): Unit = sparkTest("testIsDirExisting") {
    val dir = getTmpDir
    val configuration = sc.hadoopConfiguration

    Utils.deleteHDFSDir(dir, configuration)
    assertFalse(IOUtils.isDirExisting(dir, configuration))
    Utils.createHDFSDir(dir, configuration)
    assertTrue(IOUtils.isDirExisting(dir, configuration))
  }

  @Test
  def testProcessOutputDir(): Unit = sparkTest("testProcessOutputDir") {

    val configuration = sc.hadoopConfiguration

    // Case 1: When the output directory already exists and deleteOutputDirIfExists is true
    val dir1 = getTmpDir
    IOUtils.processOutputDir(dir1, deleteOutputDirIfExists = true, configuration)
    assertFalse(IOUtils.isDirExisting(dir1, configuration))

    // Case 2: When the output directory already exists and deleteOutputDirIfExists is false
    val dir2 = getTmpDir
    try {
      IOUtils.processOutputDir(dir2, deleteOutputDirIfExists = false, configuration)
    } catch {
      case e: Exception => assertTrue(e.isInstanceOf[IllegalArgumentException])
    } finally {
      assertTrue(IOUtils.isDirExisting(dir2, configuration))
    }

    // Case 3: When the output directory does not exist and deleteOutputDirIfExists is true
    val dir3 = getTmpDir
    Utils.deleteHDFSDir(dir3, configuration)
    IOUtils.processOutputDir(dir3, deleteOutputDirIfExists = true, configuration)
    assertFalse(IOUtils.isDirExisting(dir3, configuration))

    // Case 4: When the output directory does not exist and deleteOutputDirIfExists is false
    val dir4 = getTmpDir
    Utils.deleteHDFSDir(dir4, configuration)
    IOUtils.processOutputDir(dir4, deleteOutputDirIfExists = false, configuration)
    assertFalse(IOUtils.isDirExisting(dir4, configuration))
  }
}
