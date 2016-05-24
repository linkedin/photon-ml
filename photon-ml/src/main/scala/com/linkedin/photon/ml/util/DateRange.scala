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

import org.joda.time.LocalDate
import org.joda.time.format.DateTimeFormat

/**
 * Represents an immutable date range. In a number of places, we use date ranges as coordinates into a data set. This
 * class centralizes input processing and representations for these ranges.
 *
 * @param startDate the beginning of the range
 * @param endDate the end of the range
 */
case class DateRange(val startDate: LocalDate, val endDate: LocalDate) {

  require(!startDate.isAfter(endDate), s"Invalid range: start date $startDate comes after end date $endDate.")

  /**
   * Builds a string representation for the date range.
   *
   * @return the string representation
   */
  override def toString: String = {
    s"$startDate-$endDate"
  }

}

object DateRange {

  /**
   * Builds a new range from the start and end dates.
   *
   * @param startDate the beginning of the range
   * @param endDate the end of the range
   * @return the new date range
   */
  def fromDates(startDate: LocalDate, endDate: LocalDate): DateRange = {
    new DateRange(startDate, endDate)
  }

  /**
   * Builds a new range from string representations of the start and end dates.
   *
   * @param startDate string representing the beginning of the range
   * @param endDate string representing the end of the range
   * @return the new date range
   */
  def fromDates(startDate: String, endDate: String, pattern: String = "yyyyMMdd"): DateRange = {
    try {
      val from = DateTimeFormat.forPattern(pattern).parseLocalDate(startDate)
      val until = DateTimeFormat.forPattern(pattern).parseLocalDate(endDate)

      fromDates(from, until)

    } catch {
      case e: Exception =>
        throw new IllegalArgumentException(s"Couldn't parse the date range: $startDate-$endDate", e)
    }
  }

  /**
   * Builds a new range from string representations of the range, with start and end dates separated by "-".
   *
   * @param range string representing the range, separated by "-"
   * @return the new date range
   */
  def fromDates(range: String): DateRange = {
    val (startDate, endDate) = splitRange(range)
    fromDates(startDate, endDate)
  }

  /**
   * Builds a new range from the starting and ending days ago.
   *
   * @param startDaysAgo beginning of the range, in number of days from now
   * @param endDaysAgo end of the range, in number of days from now
   * @return the new date range
   */
  def fromDaysAgo(startDaysAgo: Int, endDaysAgo: Int, now: LocalDate = new LocalDate): DateRange = {
    require(startDaysAgo >= 0, "Start days ago cannot be negative.")
    require(endDaysAgo >= 0, "End days ago cannot be negative.")

    fromDates(now.minusDays(startDaysAgo), now.minusDays(endDaysAgo))
  }

  /**
   * Builds a new range from string representations of the starting and ending days ago.
   *
   * @param startDaysAgo string representation of the beginning of the range, in number of days from now
   * @param endDaysAgo string representation of the end of the range, in number of days from now
   * @return the new date range
   */
  def fromDaysAgo(startDaysAgo: String, endDaysAgo: String): DateRange = {
    try {
      val start = startDaysAgo.toInt
      val end = endDaysAgo.toInt

      fromDaysAgo(start, end)

    } catch {
      case nfe: NumberFormatException =>
        throw new IllegalArgumentException(
          s"Start days ago ($startDaysAgo) and end days ago ($endDaysAgo) must be valid integers.", nfe)
    }
  }

  /**
   * Builds a new range from string representations of the range, with start and end days ago separated by "-".
   *
   * @param range string representing the range, separated by "-"
   * @return the new date range
   */
  def fromDaysAgo(range: String): DateRange = {
    val (startDaysAgo, endDaysAgo) = splitRange(range)
    fromDaysAgo(startDaysAgo, endDaysAgo)
  }

  /**
   * Splits the date / day range.
   *
   * @param range the range to split
   * @param splitter the character on which to split
   * @return sequence of splits
   */
  protected def splitRange(range: String, splitter: String = "-") = {
    try {
      val Array(start, end) = range.split(splitter)
      (start, end)

    } catch {
      case e: MatchError =>
        throw new IllegalArgumentException(
          s"Couldn't parse the range: $range. Be sure to separate two values with '$splitter'.")
    }
  }

}
