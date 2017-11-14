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

import org.joda.time.{DateTime, DateTimeUtils}
import org.testng.Assert._
import org.testng.annotations.{AfterClass, BeforeClass, DataProvider, Test}

/**
 * Unit tests for [[DaysRange]].
 */
class DaysRangeTest {

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
  def invalidInputProvider(): Array[Array[Any]] = Array(
    Array(-90, 1),
    Array(90, -1),
    Array(1, 90))

  /**
   * Test that invalid date ranges are rejected.
   *
   * @param start Invalid range start
   * @param end Invalid range end
   */
  @Test(dataProvider = "invalidInputProvider", expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testInvalidInput(start: Int, end: Int): Unit = DaysRange(start, end)

  /**
   * Test that a [[DaysRange]] can be converted to a correct [[DateRange]] using today as the base date.
   */
  @Test
  def testToDateRange(): Unit = {

    val expected = DateRange.fromDateString("20160102-20160331")
    val actual = DaysRange.fromDaysString("90-1").toDateRange

    assertEquals(actual.startDate, expected.startDate)
    assertEquals(actual.endDate, expected.endDate)
  }

  @DataProvider
  def validRangeDataProvider(): Array[Array[Any]] = Array(
    Array(DaysRange.fromDaysString("90-1"), 90, 1),
    Array(DaysRange.fromDaysString("45-10"), 45, 10),
    Array(DaysRange.fromDaysString("190-25"), 190, 25))

  /**
   * Test that a [[DaysRange]] can be correctly parsed.
   *
   * @param range The parsed [[DaysRange]]
   * @param start The expected start of the range
   * @param end The expected end of the range
   */
  @Test(dataProvider = "validRangeDataProvider")
  def testFromDaysString(range: DaysRange, start: Int, end: Int): Unit = {

    assertEquals(start, range.startDays)
    assertEquals(end, range.endDays)
  }

  @DataProvider
  def invalidRangeDataProvider(): Array[Array[Any]] = Array(
    Array("90.5-1"),
    Array("words-words"))

  /**
   * Test that invalid ranges are rejected during parsing.
   *
   * @param fakeRange An invalid range as a [[String]]
   */
  @Test(dataProvider = "invalidRangeDataProvider", expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testInvalidFromDaysString(fakeRange: String): Unit = DaysRange.fromDaysString(fakeRange)

  @DataProvider
  def printRangeDataProvider(): Array[Array[Any]] = Array(
    Array(DaysRange(90, 10), None, "90-10"),
    Array(DaysRange(75, 25), Some("~/~"), "75~/~25"))

  /**
   * Test that a [[DaysRange]] can be correctly printed.
   *
   * @param daysRange The [[DaysRange]]
   * @param delimiterOpt The range delimiter
   * @param expected The expected [[String]] representation of the range
   */
  @Test(dataProvider = "printRangeDataProvider")
  def testPrintRange(
    daysRange: DaysRange,
    delimiterOpt: Option[String],
    expected: String): Unit = {

    val actual = DaysRange.printRange(daysRange, delimiterOpt.getOrElse(DaysRange.DEFAULT_DELIMITER))

    assertEquals(actual, expected)
  }
}
