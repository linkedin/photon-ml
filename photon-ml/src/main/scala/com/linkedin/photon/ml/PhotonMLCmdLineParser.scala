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
package com.linkedin.photon.ml

import com.linkedin.photon.ml.OptionNames._
import com.linkedin.photon.ml.diagnostics.DiagnosticMode
import com.linkedin.photon.ml.io.{ConstraintMapKeys, FieldNamesType, InputFormatType}
import com.linkedin.photon.ml.normalization.NormalizationType
import com.linkedin.photon.ml.optimization.{OptimizerType, RegularizationType}
import com.linkedin.photon.ml.supervised.TaskType
import scopt.OptionParser

import scala.util.parsing.json.JSON


/**
 * A collection of functions used to parse Photon-ML's parameters [[Params]]
 *
 * @author xazhang
 * @author dpeng
 * @author nkatariy
 */
object PhotonMLCmdLineParser {

  /**
   * A function that exists purely for early failure. Takes the input constraint string and only makes sure that it is
   * in the expected format. The string is expected to be an array of maps containing only the keys from
   * [[ConstraintMapKeys]]
   *
   * An example of an acceptable constraint string is
   * [
   *   {"name": "ageInHour", "term": "", "lowerBound": -1, "upperBound": 0},
   *   {"name": "ageInHour:lv", "term": "4", "lowerBound": -1},
   *   {"name": "ageInHour:lv", "term": "12", "upperBound": 0},
   *   {"name": "actor_rawclicksw.gt.0", "term": "*", "lowerBound": -0.01}
   * ]
   *
   * Please note: in addition to being called here, this function is also called from the Photon templates library to
   * validate user-given constraints.
   *
   * @param inputString constraint string
   * @return return true if input string could be successfully parsed, else return false
   */
  def checkConstraintStringValidity(inputString: String): Boolean = {
    JSON.parseFull(inputString) match {
      case Some(parsed: List[Map[String, Any]]) => true
      case _ => false
    }
  }

  /**
   * Parse [[Params]] from the command line using
   * [[http://scopt.github.io/scopt/3.3.0/api/#scopt.OptionParser scopt]]
   *
   * @param args The command line input
   * @return The parsed [[Params]]
   */
  def parseFromCommandLine(args: Array[String]): Params = {

    require(!(args.contains(s"--$TRAINING_DIAGNOSTICS") && args.contains(s"--$DIAGNOSTIC_MODE")),
      s"Specifying both $TRAINING_DIAGNOSTICS and $DIAGNOSTIC_MODE at the same time is not supported!" )

    val defaultParams = new Params()
    val params = new Params()
    val parser = new OptionParser[Unit]("Photon-ML") {
      head("Photon-ML")
      opt[String](TRAIN_DIR_OPTION)
              .required()
              .text("Input directory of training data.")
              .foreach(x => params.trainDir = x)
      opt[String](OUTPUT_DIR_OPTION)
              .required()
              .text("Photon-ML's output directory.")
              .foreach(x => params.outputDir = x)
      opt[String](TASK_TYPE_OPTION)
              .required()
              .text(s"Learning task type. Valid values: [${TaskType.values.mkString("|")}].")
              .foreach(x => params.taskType = TaskType.withName(x.toUpperCase))
      opt[String](VALIDATE_DIR_OPTION)
              .text(s"Input directory of validating data. Default: ${defaultParams.validateDirOpt}. Note that\n" +
                      s"1. Validation is optional\n" +
                      s"2. If validation data set is provided, then model validating will be performed and " +
                        s"best model will be provided\n" +
                      s"3. No matter 1 or 2, all learned models will be provided, so users can always peform " +
                        s"an independent model selection job")
              .foreach(x => params.validateDirOpt = Some(x))
      opt[Boolean](INTERCEPT_OPTION)
              .text(s"Whether to learn the intercept. Default ${defaultParams.addIntercept}.")
              .foreach(x => params.addIntercept = x)
      opt[String](REGULARIZATION_WEIGHTS_OPTION)
              .text(s"Comma separated list of regularization weights. The regularization weight will be ignored if " +
                      s"$REGULARIZATION_TYPE_OPTION is set ${RegularizationType.NONE}. " +
                      s"Default ${defaultParams.regularizationWeights.mkString(",")}.")
              .foreach(x => params.regularizationWeights = x.split(",").map(_.toDouble).toList)
      opt[String](REGULARIZATION_TYPE_OPTION)
              .text(s"The type of regularization that will be used to train the model. Options: " +
                      s"[${RegularizationType.values.mkString("|")}}]. If ${RegularizationType.NONE} is used, " +
                      s"regularization weight will be ignored. Default: ${defaultParams.regularizationType}")
              .foreach(x => params.regularizationType = RegularizationType.withName(x.toUpperCase))
      opt[Double](ELASTIC_NET_ALPHA_OPTION)
              .text(s"The alpha value in the range of [0, 1] for elastic net regularization. alpha = 1 is L1 and " +
                      s"alpha = 0 is L2. Required for elastic net regularization. Default is 0.5")
              .validate(x => if (x >= 0.0 && x <= 1.0) success else failure("alpha must be in the range [0, 1]."))
              .foreach(x => params.elasticNetAlpha = Some(x))
      opt[Int](MAX_NUM_ITERATIONS_OPTION)
              .text(s"Maximum number of iterations to run. Default ${defaultParams.maxNumIter}.")
              .foreach(x => params.maxNumIter = x)
      opt[Double](TOLERANCE_OPTION)
              .text(s"The optimizer's convergence tolerance. Default ${defaultParams.tolerance}. " +
                      s"Smaller value will lead to higher accuracy with the cost of more iterations.")
              .foreach(x => params.tolerance = x)
      opt[String](JOB_NAME_OPTION)
              .text(s"Job name of this spark application. Default: ${defaultParams.jobName}")
              .foreach(x => params.jobName = x)
      opt[String](OPTIMIZER_TYPE_OPTION)
              .text(s"The type of optimizer that will be used to train the model. Options: " +
                      s"[${OptimizerType.values.mkString("|")}}]. Default: ${defaultParams.optimizerType}}.")
              .foreach(x => params.optimizerType = OptimizerType.withName(x.toUpperCase))
      opt[Boolean](OPTIMIZATION_STATE_TRACKER_OPTION)
              .text(s"Whether to enable the optimization tracker, which tracks and stores the per-iteration log " +
                      s"information of the running optimizer Default: ${defaultParams.enableOptimizationStateTracker}")
              .foreach(x => params.enableOptimizationStateTracker = x)
      opt[Boolean](KRYO_OPTION)
              .text(s"Whether to use kryo to serialize the intermediate result, which is generally a good idea. " +
                      s"More information on serialization method used in Spark can be found through " +
                      s"the following link: https://spark.apache.org/docs/latest/tuning.html#data-serialization. " +
                        s"Default: ${defaultParams.kryo}")
              .foreach(x => params.kryo = x)
      opt[String](FORMAT_TYPE_OPTION)
              .text(s"Input Avro file's format, which contains the information of each field's name. " +
                      s"Options: [${FieldNamesType.values.mkString("|")}]. Default: ${defaultParams.fieldsNameType}")
              .foreach(x => params.fieldsNameType = FieldNamesType.withName(x.toUpperCase))
      opt[Int](MIN_NUM_PARTITIONS_OPTION)
              .text(s"The minimum number of Hadoop splits to generate.. Default: ${defaultParams.minNumPartitions}. " +
                      s"This would be potentially helpful when the number of default Hadoop splits is small. Note " +
                      s"that when the default number of Hadoop splits (from HDFS) is larger than min-partitions, " +
                      s"then min-partitions will be ignored and the number of partitions of the resulting RDD will " +
                      s"be same as the default number of Hadoop splits. In short, min-partitions will *only* be able " +
                      s"to increase the number of partitions.")
              .foreach(x => params.minNumPartitions = x)
      opt[Boolean](VALIDATE_PER_ITERATION)
              .text(s"If validating data is provided, and optimization tracker is enabled, " +
                      s"whether to compute the evaluation metrics on validating data per iteration")
              .foreach(x => params.validatePerIteration = x)
      opt[String](SUMMARIZATION_OUTPUT_DIR)
              .text("An optional directory for summarization output")
              .foreach(x => params.summarizationOutputDirOpt = Some(x))
      opt[String](NORMALIZATION_TYPE)
              .text(s"The normalization type to use in the training. Options: " +
                      s"[${NormalizationType.values().mkString("|")}]. Default: ${defaultParams.normalizationType}")
              .foreach(x => params.normalizationType = NormalizationType.valueOf(x.toUpperCase))
      opt[String](DATA_VALIDATION_TYPE)
              .text(s"The level of data validation to apply. Options: [${DataValidationType.values.mkString("|")}]. " +
                      s"Default: ${defaultParams.dataValidationType}")
              .foreach(x => params.dataValidationType = DataValidationType.withName(x.toUpperCase()))
      opt[Boolean](TRAINING_DIAGNOSTICS)
              .text(s"DEPRECATED -- USE $DIAGNOSTIC_MODE")
              .foreach(x => params.diagnosticMode = if (x) DiagnosticMode.ALL else DiagnosticMode.NONE)
      opt[String](DIAGNOSTIC_MODE)
              .text(s"Diagnostic mode after model training (${DiagnosticMode.ALL}, ${DiagnosticMode.NONE}," +
                  s" ${DiagnosticMode.TRAIN}, ${DiagnosticMode.VALIDATE}). Default: ${defaultParams.diagnosticMode}")
              .foreach(x => params.diagnosticMode = DiagnosticMode.withName(x.toUpperCase()))
      opt[String](COEFFICIENT_BOX_CONSTRAINTS)
              .text(s"JSON array of maps specifying bound constraints on coefficients. The input is expected to be " +
                      s"an array of maps with the map containing keys only from {" +
                        ConstraintMapKeys.values.mkString(s",") + "} (other " +
                      s"keys will be ignored")
              .validate(x => if (checkConstraintStringValidity(x)) {
                  success
                } else {
                  failure(s"Could not parse the input constraint string [" + x + "]. The input is expected "
                            + s"to be an array of maps with the map containing keys only from {"
                            + ConstraintMapKeys.values.mkString(",") + "}. An example input string would be: "
                            + """
                                [
                                  {"name": "ageInHour", "term": "", "lowerBound": -1, "upperBound": 0},
                                  {"name": "ageInHour:lv", "term": "4", "lowerBound": -1},
                                  {"name": "ageInHour:lv", "term": "12", "upperBound": 0},
                                  {"name": "actor_rawclicksw.gt.0", "term": "*", "lowerBound": -0.01}
                                ]
                            """
                  )
                }
              )
              .foreach(x => params.constraintString = Some(x))
      opt[String](SELECTED_FEATURES_FILE)
              .text(s"Path to the file containing the features to be selected for training")
              .foreach(x => params.selectedFeaturesFile = Some(x))
      opt[Int](TREE_AGGREGATE_DEPTH)
        .text("The depth of the aggregate tree used in treeAggregate for the loss function. When the depth is 1, it " +
              "works as normal linear aggregate. A depth larger than 1 consumes less memory in driver side and is " +
              "potentially faster. Be aware that treeAggregate with depth > 1 is unstable and may be slow in Spark " +
              "1.4 and 1.5. We recommend setting depth = 2 when tree structure aggregation is needed. " +
              s"Default: ${defaultParams.treeAggregateDepth}")
        .validate(x => if (x > 0) {
            success
          } else {
            failure(s"$TREE_AGGREGATE_DEPTH has to be a positive integer. Your input is $x")
          }
        )
        .foreach(x => params.treeAggregateDepth = x)
      opt[String](OFFHEAP_INDEXMAP_DIR)
        .text("The offheap storage directory if offheap map is needed. DefaultIndexMap will be used if not specified.")
        .foreach(x => params.offHeapIndexMapDir = Some(x))
      opt[Int](OFFHEAP_INDEXMAP_NUM_PARTITIONS)
        .text("The number of partitions for the offheap map storage. This partition number should be consistent with " +
            "the number when offheap storage is built. This parameter affects only the execution speed during " +
            "feature index building and has zero performance impact on training other than maintaining a " +
            "convention.")
        .foreach(x => params.offHeapIndexMapNumPartitions = x)
      opt[Boolean](DELETE_OUTPUT_DIRS_IF_EXIST)
        .text(s"Delete the output directories (including the model and summarization output directories) if exist." +
            s"Default: ${defaultParams.deleteOutputDirsIfExist}")
        .foreach(x => params.deleteOutputDirsIfExist = x)
      opt[String](INPUT_FILE_FORMAT)
        .text("Indicating the input data format for PhotonML")
        .foreach(x => params.inputFormatType = InputFormatType.withName(x.toUpperCase()))
      opt[Int](FEATURE_DIMENSION)
        .text("A preliminary indicator of how many features the input data contains (just an upper bound, not including intercept). Only used by LIBSVM foramt and is to be removed soon.")
        .foreach(x => params.featureDimension = x)
      help(HELP_OPTION).text("prints Photon-ML's usage text")
      override def showUsageOnError = true
    }
    if (!parser.parse(args)) {
      throw new IllegalArgumentException(s"Parsing the command line arguments failed.\n" +
                                         s"Input arguments are: ${args.mkString(", ")}).")
    }
    params.validate()
    postProcess(params)
    params
  }

  /**
   * This method post process params for necessary value override.
   *
   * @param inputParams Parsed params
   * @return Post processed params
   */
  private def postProcess(inputParams: Params): Unit = {
    // Append zero regularization weight if there is no regularization set
    if (inputParams.regularizationType == RegularizationType.NONE) {
      inputParams.regularizationWeights = List(0.0)
    }
  }
}
