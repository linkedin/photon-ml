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

import org.joda.time.{DateTime, DateTimeUtils, LocalDate}
import org.testng.Assert._
import org.testng.annotations.{AfterClass, BeforeClass, DataProvider, Test}

/**
 * Unit tests for [[DateRange]].
 */
class DateRangeTest {

  @DataProvider
  def rangeDataProvider(): Array[Array[Any]] = Array(
    Array(DateRange.fromDateString("20150101-20150101"), "2015-01-01", "2015-01-01"),
    Array(DateRange.fromDateString("20150101-20150201"), "2015-01-01", "2015-02-01"),
    Array(DateRange.fromDateString("20140312-20150211"), "2014-03-12", "2015-02-11"),
    Array(DateRange.fromDateString("19950816-20011120"), "1995-08-16", "2001-11-20"))

  /**
   * Test that a [[DateRange]] can be correctly parsed.
   *
   * @param range The parsed [[DateRange]]
   * @param startDate The expected start of the range
   * @param endDate The expected end of the range
   */
  @Test(dataProvider = "rangeDataProvider")
  def testFromDateString(range: DateRange, startDate: String, endDate: String): Unit = {
    val start = LocalDate.parse(startDate)
    val end = LocalDate.parse(endDate)

    assertEquals(start, range.startDate)
    assertEquals(end, range.endDate)
  }

  @DataProvider
  def invalidRangeDataProvider(): Array[Array[Any]] = Array(
    Array("20160101-19551105"),
    Array("words-words"))

  /**
   * Test that invalid ranges are rejected during parsing.
   *
   * @param fakeRange An invalid range as a [[String]]
   */
  @Test(dataProvider = "invalidRangeDataProvider", expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testInvalidFromDateString(fakeRange: String): Unit = DateRange.fromDateString(fakeRange)

  @DataProvider
  def validSplitRangeData(): Array[Array[Any]] = Array(
    Array("abc_-xdef", Some("_"), "abc", "-xdef"),
    Array("abc_-xdef", Some("x"), "abc_-", "def"),
    Array("abc_-xdef", Some("_-x"), "abc", "def"),
    Array("abc_-xdef", None, "abc_", "xdef"))

  /**
   * Test that ranges can be split correctly, regardless of delimiter.
   *
   * @param fakeRange Some range
   * @param delimiterOpt An optional delimiter to use
   * @param start The expected start of the range
   * @param end The expected end of the range
   */
  @Test(dataProvider = "validSplitRangeData")
  def testValidSplitRange(fakeRange: String, delimiterOpt: Option[String], start: String, end: String): Unit = {
    val rangePair = delimiterOpt match {
      case Some(delimiter) => DateRange.splitRange(fakeRange, delimiter)
      case None => DateRange.splitRange(fakeRange)
    }

    assertEquals(rangePair._1, start)
    assertEquals(rangePair._2, end)
  }

  @DataProvider
  def invalidSplitRangeData(): Array[Array[Any]] = Array(
    Array("abc_def_ghi", "_"),
    Array("abc", "_"))

  /**
   * Test that incorrectly delimited ranges will be rejected.
   *
   * @param fakeRange Some range
   * @param delimiter The delimiter to use
   */
  @Test(dataProvider = "invalidSplitRangeData", expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testInvalidSplitRange(fakeRange: String, delimiter: String): Unit = {
    DateRange.splitRange(fakeRange, delimiter)
  }

  @DataProvider
  def printRangeDataProvider(): Array[Array[Any]] = Array(
    Array(DateRange.fromDateString("20150101-20150101"), None, None, "20150101-20150101"),
    Array(DateRange.fromDateString("20150101-20150101"), Some("yyyy-MM-dd"), None, "2015-01-01-2015-01-01"),
    Array(DateRange.fromDateString("20150101-20150101"), None, Some("~/~"), "20150101~/~20150101"),
    Array(DateRange.fromDateString("20150101-20150101"), Some("yyyy-MM-dd"), Some("~/~"), "2015-01-01~/~2015-01-01"))

  /**
   * Test that a [[DateRange]] can be correctly printed.
   *
   * @param dateRange The [[DateRange]]
   * @param delimiterOpt The range delimiter
   * @param patternOpt\
   * @param expected The expected [[String]] representation of the range
   *
   */
  @Test(dataProvider = "printRangeDataProvider")
  def testPrintRange(
      dateRange: DateRange,
      patternOpt: Option[String],
      delimiterOpt: Option[String],
      expected: String): Unit = {

    val actual = DateRange.printRange(
      dateRange,
      patternOpt.getOrElse(DateRange.DEFAULT_PATTERN),
      delimiterOpt.getOrElse(DateRange.DEFAULT_DELIMITER))

    assertEquals(actual, expected)
  }
}
