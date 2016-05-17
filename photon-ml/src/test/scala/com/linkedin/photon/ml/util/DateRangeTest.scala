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

import org.joda.time.DateTime
import org.joda.time.DateTimeUtils
import org.joda.time.LocalDate
import org.joda.time.LocalDate
import org.testng.annotations.{AfterClass, BeforeClass, DataProvider, Test}
import org.testng.Assert._

/**
 * This class tests [[DateRange]].
 */
class DateRangeTest {

  val today = "2016-04-01"

  @BeforeClass
  def setup() {
    DateTimeUtils.setCurrentMillisFixed(DateTime.parse(today).getMillis());
  }

  @AfterClass
  def teardown() {
    DateTimeUtils.setCurrentMillisSystem();
  }

  @DataProvider
  def rangeDataProvider(): Array[Array[Any]] = {
    Array(
      Array(DateRange.fromDates("20150101-20150201"), "2015-01-01", "2015-02-01"),
      Array(DateRange.fromDates("20140312-20150211"), "2014-03-12", "2015-02-11"),
      Array(DateRange.fromDates("19950816-20011120"), "1995-08-16", "2001-11-20"),
      Array(DateRange.fromDaysAgo("90-1"), "2016-01-02", "2016-03-31"),
      Array(DateRange.fromDaysAgo("45-10"), "2016-02-16", "2016-03-22"),
      Array(DateRange.fromDaysAgo("190-25"), "2015-09-24", "2016-03-07")
    )
  }

  @Test(dataProvider = "rangeDataProvider")
  def testFromRange(range: DateRange, startDate: String, endDate: String): Unit = {
    val start = LocalDate.parse(startDate)
    val end = LocalDate.parse(endDate)

    assertEquals(start, range.startDate)
    assertEquals(end, range.endDate)
  }

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testStartDateAfterEndDate() {
    DateRange.fromDates("20160101-19551105")
  }

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testInvalidRange() {
    DateRange.fromDates("blahblah")
  }

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testInvalidRangeDates() {
    DateRange.fromDates("blah-blah")
  }

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testInvalidDaysAgo() {
    DateRange.fromDaysAgo("blahblah")
  }

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testInvalidDaysAgoDays() {
    DateRange.fromDaysAgo("blah-blah")
  }

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testNegativeDaysAgoDays() {
    DateRange.fromDaysAgo("-90-1")
  }
}
