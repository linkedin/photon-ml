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

import scala.util.Try

import org.joda.time.LocalDate
import org.joda.time.format.DateTimeFormat

/**
 * Represents an immutable date range.
 *
 * @param startDate The beginning of the range
 * @param endDate The end of the range
 */
case class DateRange(startDate: LocalDate, endDate: LocalDate) {

  import DateRange._

  require(!startDate.isAfter(endDate), s"Invalid range: start date $startDate comes after end date $endDate.")

  override def toString: String = printRange(this)
}

object DateRange {

  val DEFAULT_PATTERN: String = "yyyyMMdd"
  val DEFAULT_DELIMITER: String = "-"

  /**
   * Builds a new range from string representations of the start and end dates.
   *
   * @param startDate String representing the beginning of the range
   * @param endDate String representing the end of the range
   * @return The new date range
   */
  def fromDateStrings(startDate: String, endDate: String, pattern: String = DEFAULT_PATTERN): DateRange = {
    try {
      val dateTimeFormatter = DateTimeFormat.forPattern(pattern)
      val from = dateTimeFormatter.parseLocalDate(startDate)
      val until = dateTimeFormatter.parseLocalDate(endDate)

      DateRange(from, until)

    } catch {
      case e: Exception =>
        throw new IllegalArgumentException(s"Couldn't parse the date range: $startDate-$endDate", e)
    }
  }

  /**
   * Builds a new [[DateRange]] from a string representations of the range, with start and end dates separated by a
   * delimiter.
   *
   * @param range String representing the range, separated by a delimiter
   * @return The new date range
   */
  def fromDateString(range: String): DateRange = {

    val (startDate, endDate) = splitRange(range)
    fromDateStrings(startDate, endDate)
  }

  /**
   * Splits the date / day range.
   *
   * @param range The range to split
   * @param delimiter The character on which to split
   * @return Sequence of splits
   */
  def splitRange(range: String, delimiter: String = DEFAULT_DELIMITER): (String, String) =
    Try {
        val Array(start, end) = range.split(delimiter)
        (start, end)
      }
      .getOrElse(throw new IllegalArgumentException(s"Couldn't parse the range '$range' using delimiter '$delimiter'."))

  /**
   * Write an input [[DateRange]] to a [[String]].
   *
   * @param dateRange The input date range
   * @param pattern The pattern to use for the date
   * @param delimiter The delimiter between the start and end dates
   * @return A [[String]] representation of the input [[DateRange]]
   */
  def printRange(
      dateRange: DateRange,
      pattern: String = DEFAULT_PATTERN,
      delimiter: String = DEFAULT_DELIMITER): String = {

    val dateTimeFormatter = DateTimeFormat.forPattern(pattern)

    s"${dateTimeFormatter.print(dateRange.startDate)}$delimiter${dateTimeFormatter.print(dateRange.endDate)}"
  }
}
