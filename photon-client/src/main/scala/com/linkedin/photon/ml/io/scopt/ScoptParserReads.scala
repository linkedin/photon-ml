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
package com.linkedin.photon.ml.io.scopt

import org.apache.hadoop.fs.Path
import org.joda.time.DateTimeZone

import com.linkedin.photon.ml.{DataValidationType, HyperparameterTunerName, HyperparameterTuningMode, TaskType}
import com.linkedin.photon.ml.DataValidationType.DataValidationType
import com.linkedin.photon.ml.HyperparameterTunerName.HyperparameterTunerName
import com.linkedin.photon.ml.HyperparameterTuningMode.HyperparameterTuningMode
import com.linkedin.photon.ml.TaskType.TaskType
import com.linkedin.photon.ml.evaluation.EvaluatorType
import com.linkedin.photon.ml.io.ModelOutputMode
import com.linkedin.photon.ml.io.ModelOutputMode.ModelOutputMode
import com.linkedin.photon.ml.normalization.NormalizationType
import com.linkedin.photon.ml.normalization.NormalizationType.NormalizationType
import com.linkedin.photon.ml.util.{DateRange, DaysRange, Utils}

/**
 * Implicit Scopt [[scopt.Read]] objects, used to read Scopt parameters directly into additional non-string types.
 */
object ScoptParserReads {

  implicit val dataValidationTypeRead: scopt.Read[DataValidationType] = scopt.Read.reads(DataValidationType.withName)
  implicit val dateRangeRead: scopt.Read[DateRange] = scopt.Read.reads(DateRange.fromDateString)
  implicit val daysRangeRead: scopt.Read[DaysRange] = scopt.Read.reads(DaysRange.fromDaysString)
  implicit val evaluatorTypeRead: scopt.Read[EvaluatorType] = scopt.Read.reads(Utils.evaluatorWithName)
  implicit val hyperParameterTunerNameRead: scopt.Read[HyperparameterTunerName] =
    scopt.Read.reads(HyperparameterTunerName.withName)
  implicit val hyperParameterTuningModeRead: scopt.Read[HyperparameterTuningMode] =
    scopt.Read.reads(HyperparameterTuningMode.withName)
  implicit val modelOutputModeRead: scopt.Read[ModelOutputMode] = scopt.Read.reads(ModelOutputMode.withName)
  implicit val normalizationTypeRead: scopt.Read[NormalizationType] = scopt.Read.reads(NormalizationType.withName)
  implicit val pathRead: scopt.Read[Path] = scopt.Read.reads(parsePath)
  implicit val taskTypeRead: scopt.Read[TaskType] = scopt.Read.reads(TaskType.withName)

  // For a list of valid timezone ids, see:
  // http://joda-time.sourceforge.net/timezones.html
  implicit val timeZoneRead: scopt.Read[DateTimeZone] = scopt.Read.reads(DateTimeZone.forID)

  /**
   * Create a HDFS path to a file/directory defined by a [[String]].
   *
   * @param input The path to a HDFS file/directory, as a [[String]]
   * @return A [[Path]] to the file/directory
   */
  private def parsePath(input: String): Path = new Path(input)
}
