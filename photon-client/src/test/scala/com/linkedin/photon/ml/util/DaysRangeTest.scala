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

import org.joda.time.{DateTime, DateTimeUtils, DateTimeZone, LocalDate}
import org.testng.Assert._
import org.testng.annotations.{AfterClass, BeforeClass, DataProvider, Test}

/**
 * Unit tests for [[DaysRange]].
 */
class DaysRangeTest {

  // 4/2 in UTC, 4/1 in America/Los_Angeles
  private val today = new DateTime(2016, 4, 2, 1, 0, DateTimeZone.UTC)

  @BeforeClass
  def setup() {
    DateTimeUtils.setCurrentMillisFixed(today.getMillis)
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

    val expected = DateRange.fromDateString("20160103-20160401")

    // Today is 4/2 in UTC. One day ago should be 4/1 in UTC
    val actual = DaysRange.fromDaysString("90-1").toDateRange()

    assertEquals(actual.startDate, expected.startDate)
    assertEquals(actual.endDate, expected.endDate)
  }

  /**
   * Test that a [[DaysRange]] can be converted to a correct [[DateRange]] using today as the base date in PST.
   */
  @Test
  def testToDateRangePst(): Unit = {

    val expected = DateRange.fromDateString("20160102-20160331")

    // Today is 4/2 in UTC and 4/1 in America/Los_Angeles. One day ago should be 3/31 in America/Los_Angeles
    val actual = DaysRange.fromDaysString("90-1").toDateRange(DateTimeZone.forID("America/Los_Angeles"))

    assertEquals(actual.startDate, expected.startDate)
    assertEquals(actual.endDate, expected.endDate)
  }

  @DataProvider
  def validRangeDataProvider(): Array[Array[Any]] = Array(
    Array(DaysRange.fromDaysString("0-0"), 0, 0),
    Array(DaysRange.fromDaysString("90-0"), 90, 0),
    Array(DaysRange.fromDaysString("90-89"), 90, 89))

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
