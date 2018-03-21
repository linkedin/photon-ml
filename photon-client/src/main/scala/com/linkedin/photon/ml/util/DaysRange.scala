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

import org.joda.time.LocalDate

/**
 * Represents an immutable date range, in days prior to today.
 *
 * @param startDays The beginning of the range
 * @param endDays The end of the range
 */
case class DaysRange(startDays: Int, endDays: Int) {

  import DaysRange._

  require(startDays >= 0, s"Invalid range: # days for start of range '$startDays' is less than 0.")
  require(endDays >= 0, s"Invalid range: # days for end of range '$endDays' is less than 0.")
  require(
    startDays >= endDays,
    s"Invalid range: start of range '$startDays' is fewer days ago than end of range '$endDays'.")

  /**
   * Convert the [[DaysRange]] to a [[DateRange]].
   *
   * @return The [[DateRange]] equivalent of the [[DaysRange]] at the time of function call
   */
  def toDateRange: DateRange = {

    val now: LocalDate = LocalDate.now

    DateRange(now.minusDays(startDays), now.minusDays(endDays))
  }

  override def toString: String = printRange(this)
}

object DaysRange {

  val DEFAULT_DELIMITER: String = DateRange.DEFAULT_DELIMITER

  /**
   * Builds a new [[DaysRange]] from a string representations of the range, with start and end dates separated by a
   * delimiter.
   *
   * @param range String representing the range, separated by a delimiter
   * @return The new date range
   */
  def fromDaysString(range: String): DaysRange = {

    val (startDay, endDay) = DateRange.splitRange(range, DEFAULT_DELIMITER)

    DaysRange(startDay.toInt, endDay.toInt)
  }

  /**
   * Write an input [[DaysRange]] to a [[String]].
   *
   * @param daysRange The input date range
   * @param delimiter The delimiter between the start and end dates
   * @return A [[String]] representation of the input [[DaysRange]]
   */
  def printRange(daysRange: DaysRange, delimiter: String = DEFAULT_DELIMITER): String =
    s"${daysRange.startDays}$delimiter${daysRange.endDays}"
}
