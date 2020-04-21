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

import com.linkedin.photon.ml.HyperparameterTunerName.HyperparameterTunerName
import com.linkedin.photon.ml.HyperparameterTuningMode.HyperparameterTuningMode
import com.linkedin.photon.ml.TaskType.TaskType
import com.linkedin.photon.ml.Types.CoordinateId
import com.linkedin.photon.ml.cli.game.training.GameTrainingDriver
import com.linkedin.photon.ml.io.ModelOutputMode.ModelOutputMode
import com.linkedin.photon.ml.io.scopt.{ScoptParameter, ScoptParserHelpers, ScoptParserReads}
import com.linkedin.photon.ml.io.{CoordinateConfiguration, ModelOutputMode}
import com.linkedin.photon.ml.normalization.NormalizationType
import com.linkedin.photon.ml.normalization.NormalizationType.NormalizationType
import com.linkedin.photon.ml.optimization.VarianceComputationType
import com.linkedin.photon.ml.optimization.VarianceComputationType.VarianceComputationType
import com.linkedin.photon.ml.util.{DateRange, DaysRange}
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
        additionalDocs = Seq(s"training tasks: ${TaskType.values.mkString(", ")}"),
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

      // Hyper Parameter Tuner Name
      ScoptParameter[HyperparameterTunerName, HyperparameterTunerName](
        GameTrainingDriver.hyperParameterTunerName,
        usageText = "<type>"),

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
      ScoptParameter[VarianceComputationType, VarianceComputationType](
        GameTrainingDriver.varianceComputationType,
        additionalDocs = Seq(s"options: ${VarianceComputationType.values.map(_.toString).mkString(", ")}")),

      // Model Sparsity Threshold
      ScoptParameter[Double, Double](
        GameTrainingDriver.modelSparsityThreshold),

      // Ignore Threshold for New Models
      ScoptParameter[Boolean, Boolean](
        GameTrainingDriver.ignoreThresholdForNewModels),

      // Incremental training
      ScoptParameter[Boolean, Boolean](
        GameTrainingDriver.incrementalTraining),

      // Save per-group evaluation result
      ScoptParameter[Boolean, Boolean](
        GameTrainingDriver.savePerGroupEvaluationResult))

  override protected val parser: OptionParser[ParamMap] = new OptionParser[ParamMap]("GAME-Training") {

    /**
     * Helper method to convert a [[ScoptParameter]] object to a defined Scopt parameter object.
     *
     * @tparam In The type of the command line argument when parsed by Scopt
     * @param scoptParameter A Scopt wrapper for a [[org.apache.spark.ml.param.Param]] which contains extra information
     *                       to define how to parse/print it from/to the command line
     * @return A Scopt [[OptionDef]]
     */
    private def optHelper[In, Out](scoptParameter: ScoptParameter[In, Out]): OptionDef[In, ParamMap] = {

      implicit val read: Read[In] = scoptParameter.read

      scoptParameter.toOptionDef(opt[In])
    }

    scoptGameTrainingParams.foreach(optHelper(_))
  }

  /**
   * Convert parameters stored in a valid [[ParamMap]] object to [[String]] format for output to the command line, in a
   * format which can be parsed back into a valid [[ParamMap]].
   *
   * @param paramMap A valid [[ParamMap]]
   * @return A [[Seq]] of [[String]] representations of the parameters, in a format that can be parsed by Scopt
   */
  override def printForCommandLine(paramMap: ParamMap): Seq[String] = {

    GameTrainingDriver.validateParams(paramMap)

    scoptGameTrainingParams.flatMap(_.generateCmdLineArgs(paramMap))
  }
}
