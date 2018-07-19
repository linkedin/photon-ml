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
package com.linkedin.photon.ml.io.scopt.game

import scala.language.existentials

import org.apache.hadoop.fs.Path
import org.apache.spark.ml.param.ParamMap
import scopt.{OptionDef, OptionParser, Read}

import com.linkedin.photon.ml.HyperparameterTuningMode.HyperparameterTuningMode
import com.linkedin.photon.ml.TaskType.TaskType
import com.linkedin.photon.ml.Types.CoordinateId
import com.linkedin.photon.ml.cli.game.training.GameTrainingDriver
import com.linkedin.photon.ml.io.ModelOutputMode.ModelOutputMode
import com.linkedin.photon.ml.io.scopt.{ScoptParameter, ScoptParserHelpers, ScoptParserReads}
import com.linkedin.photon.ml.io.{CoordinateConfiguration, ModelOutputMode}
import com.linkedin.photon.ml.normalization.NormalizationType
import com.linkedin.photon.ml.normalization.NormalizationType.NormalizationType
import com.linkedin.photon.ml.util.{DateRange, DaysRange, DoubleRange}
import com.linkedin.photon.ml.{HyperparameterTuningMode, TaskType}

/**
 * Scopt command line argument parser for GAME training parameters.
 */
object ScoptGameTrainingParametersParser extends ScoptGameParametersParser {

  import ScoptParserReads._

  val scoptGameTrainingParams: Seq[ScoptParameter[In, Out] forSome { type In; type Out }] =
    createScoptGameParams(GameTrainingDriver) ++ Seq(

      // Model Input Dir
      ScoptParameter[Path, Path](
        GameTrainingDriver.modelInputDirectory,
        usageText = "<path>",
        isRequired = false),

      // Task Type
      ScoptParameter[TaskType, TaskType](
        GameTrainingDriver.trainingTask,
        usageText = s"<task>",
        additionalDocs = Seq(s"training tasks: ${TaskType.values.filterNot(_ == TaskType.NONE).mkString(", ")}"),
        isRequired = true),

      // Validation Data Directories
      ScoptParameter[Seq[Path], Set[Path]](
        GameTrainingDriver.validationDataDirectories,
        parse = ScoptParserHelpers.parseSetFromSeq,
        print = ScoptParserHelpers.iterableToString,
        usageText = "<path1>,<path2>,..."),

      // Validation Data Date Range
      ScoptParameter[DateRange, DateRange](
        GameTrainingDriver.validationDataDateRange,
        usageText = s"${DateRange.DEFAULT_PATTERN}${DateRange.DEFAULT_DELIMITER}${DateRange.DEFAULT_PATTERN}"),

      // Validation Data Days Range
      ScoptParameter[DaysRange, DaysRange](
        GameTrainingDriver.validationDataDaysRange,
        usageText = s"xx${DaysRange.DEFAULT_DELIMITER}xx"),

      // Minimum Validation Partitions
      ScoptParameter[Int, Int](
        GameTrainingDriver.minValidationPartitions,
        usageText = "<value>"),

      // Partial Retraining Locked Coordinates
      ScoptParameter[Seq[CoordinateId], Set[CoordinateId]](
        GameTrainingDriver.partialRetrainLockedCoordinates,
        parse = ScoptParserHelpers.parseSetFromSeq,
        print = ScoptParserHelpers.iterableToString,
        usageText = "<coordinate1>,<coordinate2>,..."),

      // Output Mode
      ScoptParameter[ModelOutputMode, ModelOutputMode](
        GameTrainingDriver.outputMode,
        usageText = "<mode>",
        additionalDocs = Seq(s"output modes: ${ModelOutputMode.values.map(_.toString).mkString(", ")}")),

      // Coordinate Configurations
      ScoptParameter[Map[String, String], Map[CoordinateId, CoordinateConfiguration]](
        GameTrainingDriver.coordinateConfigurations,
        parse = ScoptParserHelpers.parseCoordinateConfiguration,
        updateOpt = Some(ScoptParserHelpers.updateCoordinateConfigurations),
        printSeq = ScoptParserHelpers.coordinateConfigsToStrings,
        usageText = "<arg>=<value>",
        additionalDocs = Seq(
          s"required args: ${ScoptParserHelpers.formatArgs(ScoptParserHelpers.COORDINATE_CONFIG_REQUIRED_ARGS)}",
          s"random effect required args: ${ScoptParserHelpers.formatArgs(ScoptParserHelpers.COORDINATE_CONFIG_RANDOM_EFFECT_REQUIRED_ARGS)}",
          s"optional args: ${ScoptParserHelpers.formatArgs(ScoptParserHelpers.COORDINATE_CONFIG_OPTIONAL_ARGS)}",
          s"fixed effect optional args: ${ScoptParserHelpers.formatArgs(ScoptParserHelpers.COORDINATE_CONFIG_FIXED_EFFECT_OPTIONAL_ARGS)}",
          s"random effect optional args: ${ScoptParserHelpers.formatArgs(ScoptParserHelpers.COORDINATE_CONFIG_RANDOM_EFFECT_OPTIONAL_ARGS)}"),
        isRequired = true),

      // Coordinate Update Sequence
      ScoptParameter[Seq[CoordinateId], Seq[CoordinateId]](
        GameTrainingDriver.coordinateUpdateSequence,
        print = ScoptParserHelpers.iterableToString,
        usageText = "<coordinate1>,<coordinate2>,...",
        isRequired = true),

      // Coordinate Descent Iterations
      ScoptParameter[Int, Int](
        GameTrainingDriver.coordinateDescentIterations,
        usageText = "<value>",
        isRequired = true),

      // Normalization
      ScoptParameter[NormalizationType, NormalizationType](
        GameTrainingDriver.normalization,
        usageText = "<type>",
        additionalDocs = Seq(s"output modes: ${NormalizationType.values.map(_.toString).mkString(", ")}")),

      // Data Summary Directory
      ScoptParameter[Path, Path](
        GameTrainingDriver.dataSummaryDirectory,
        usageText = "<path>"),

      // Tree Aggregate Depth
      ScoptParameter[Int, Int](
        GameTrainingDriver.treeAggregateDepth,
        usageText = "<value>"),

      // Hyper Parameter Tuning
      ScoptParameter[HyperparameterTuningMode, HyperparameterTuningMode](
        GameTrainingDriver.hyperParameterTuning,
        usageText = "<type>",
        additionalDocs = Seq(s"output modes: ${HyperparameterTuningMode.values.map(_.toString).mkString(", ")}")),

      // Hyper Parameter Tuning Iterations
      ScoptParameter[Int, Int](
        GameTrainingDriver.hyperParameterTuningIter,
        usageText = "<value>"),

      // Compute Variance
      ScoptParameter[Boolean, Boolean](
        GameTrainingDriver.computeVariance),

      // Model Sparsity Threshold
      ScoptParameter[Double, Double](
        GameTrainingDriver.modelSparsityThreshold),

      // Ignore Threshold for New Models
      ScoptParameter[Boolean, Boolean](
        GameTrainingDriver.ignoreThresholdForNewModels))

  /**
   * Parse command line arguments for GAME training into a [[ParamMap]].
   *
   * @param args [[Array]] of command line arguments
   * @return An initialized [[ParamMap]]
   */
  def parseFromCommandLine(args: Array[String]): ParamMap = {

    val parser = new OptionParser[ParamMap]("GAME-Training") {

      private def optHelper[In, Out](scoptParameter: ScoptParameter[In, Out]): OptionDef[In, ParamMap] = {

        implicit val read: Read[In] = scoptParameter.read

        scoptParameter.toOptionDef(opt[In])
      }

      scoptGameTrainingParams.foreach(optHelper(_))
    }

    parser.parse(args, ParamMap.empty) match {
      case Some(params) => params

      case None =>
        val errMsg = args
          .grouped(2)
          .map(_.mkString(" "))
          .mkString("\n")

        throw new IllegalArgumentException(s"Parsing the following command line arguments failed:\n${errMsg.toString}")
    }
  }

  /**
   * Given a [[ParamMap]] of valid parameters, convert them into a [[Seq]] of [[String]] representations which can be
   * parsed by Scopt.
   *
   * @param paramMap Valid GAME training parameters
   * @return A [[Seq]] of [[String]] representations of the parameters, in a format that can be parsed by Scopt
   */
  def printForCommandLine(paramMap: ParamMap): Seq[String] = {

    GameTrainingDriver.validateParams(paramMap)

    scoptGameTrainingParams.flatMap(_.generateCmdLineArgs(paramMap))
  }
}
