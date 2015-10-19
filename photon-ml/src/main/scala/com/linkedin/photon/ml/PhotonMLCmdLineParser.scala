package com.linkedin.photon.ml

import OptionNames._
import com.linkedin.photon.ml.io.FieldNamesType
import com.linkedin.photon.ml.optimization.RegularizationType
import com.linkedin.photon.ml.supervised.TaskType
import TaskType._
import com.linkedin.photon.ml.io.{ConstraintMapKeys, FieldNamesType}
import com.linkedin.photon.ml.normalization.NormalizationType
import com.linkedin.photon.ml.optimization.{OptimizerType, RegularizationType}
import com.linkedin.photon.ml.supervised.TaskType
import scopt.OptionParser

import scala.util.parsing.json.JSON


/**
 * A collection of functions used to parse MLEase's parameters [[Params]]
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
   * @param inputString constraint string
   * @return return true if input string could be successfully parsed, else return false
   */
  private def checkConstraintStringValidity(inputString: String): Boolean = {
    JSON.parseFull(inputString) match {
      case Some(parsed: List[Map[String, Any]]) => true
      case None => false
    }
  }

  /**
   * Parse [[Params]] from the command line using
   * [[http://scopt.github.io/scopt/3.3.0/api/#scopt.OptionParser scopt]]
   * @param args The command line input
   * @return The parsed [[Params]]
   */
  def parseFromCommandLine(args: Array[String]): Params = {

    val defaultParams = Params()
    val parser = new OptionParser[Params]("ML-Ease") {
      head("ML-Ease")
      opt[String](TRAIN_DIR_OPTION)
        .required()
        .text("Input directory of training data.")
        .action((x, c) => c.copy(trainDir = x))
      opt[String](OUTPUT_DIR_OPTION)
        .required()
        .text("MLEase's output directory.")
        .action((x, c) => c.copy(outputDir = x))
      opt[String](TASK_TYPE_OPTION)
        .required()
        .text(s"Learning task type, e.g., $LINEAR_REGRESSION, $POISSON_REGRESSION or $LOGISTIC_REGRESSION.")
        .action((x, c) => c.copy(taskType = TaskType.withName(x.toUpperCase)))
      opt[String](VALIDATE_DIR_OPTION)
        .text(s"Input directory of validating data. Default: ${defaultParams.validateDirOpt}. Note that\n" +
        s"1. Validation is optional\n" +
        s"2. If validation data set is provided, then model validating will be performed and best model will be provided\n" +
        s"3. No matter 1 or 2, all learned models will be provided, so users can always peform an independent model selection job")
        .action((x, c) => c.copy(validateDirOpt = Some(x)))
      opt[Boolean](INTERCEPT_OPTION)
        .text(s"Whether to learn the intercept. Default ${defaultParams.addIntercept}.")
        .action((x, c) => c.copy(addIntercept = x))
      opt[String](REGULARIZATION_WEIGHTS_OPTION)
        .text(s"Comma separated list of regularization weights. The regularization weight will be ignored if $REGULARIZATION_TYPE_OPTION is set ${RegularizationType.NONE}. " +
              s"Default ${defaultParams.regularizationWeights.mkString(",")}.")
        .action((x, c) => c.copy(regularizationWeights = x.split(",").map(_.toDouble).toList))
      opt[String](REGULARIZATION_TYPE_OPTION)
        .text(s"The type of regularization that will be used to train the model. Options: [${RegularizationType.values.mkString("|")}}]. If ${RegularizationType.NONE} is used, " +
              s"regularization weight will be ignored. Default: ${defaultParams.regularizationType}")
        .action((x, c) => c.copy(regularizationType = RegularizationType.withName(x.toUpperCase)))
      opt[Double](ELASTIC_NET_ALPHA_OPTION)
        .text(s"The alpha value in the range of [0, 1] for elastic net regularization. alpha = 1 is L1 and alpha = 0 is L2. Required for elastic net regularization.")
        .validate(x => if (x >= 0.0 && x <= 1.0) success else failure("alpha must be in the range [0, 1]."))
        .action((x, c) => c.copy(elasticNetAlpha = Some(x)))
      opt[Int](MAX_NUM_ITERATIONS_OPTION)
        .text(s"Maximum number of iterations to run. Default ${defaultParams.maxNumIter}.")
        .action((x, c) => c.copy(maxNumIter = x))
      opt[Double](TOLERANCE_OPTION)
        .text(s"The optimizer's convergence tolerance. Default ${defaultParams.tolerance}. " +
        s"Smaller value will lead to higher accuracy with the cost of more iterations.")
        .action((x, c) => c.copy(tolerance = x))
      opt[String](JOB_NAME_OPTION)
        .text(s"Job name of this spark application. Default: ${defaultParams.jobName}")
        .action((x, c) => c.copy(jobName = x))
      opt[String](OPTIMIZER_TYPE_OPTION)
        .text(s"The type of optimizer that will be used to train the model. Options: [${OptimizerType.values.mkString("|")}}]. " +
        s"Default: ${defaultParams.optimizerType}}.")
        .action((x, c) => c.copy(optimizerType = OptimizerType.withName(x.toUpperCase)))
      opt[Boolean](OPTIMIZATION_STATE_TRACKER_OPTION)
        .text(s"Whether to enable the optimization tracker, which tracks and stores the per-iteration log information " +
        s"of the running optimizer Default: ${defaultParams.enableOptimizationStateTracker}")
        .action((x, c) => c.copy(enableOptimizationStateTracker = x))
      opt[Boolean](KRYO_OPTION)
        .text(s"Whether to use kryo to serialize the intermediate result, which is generally a good idea. " +
        s"More information on serialization method used in Spark can be found through " +
        s"the following link: https://spark.apache.org/docs/latest/tuning.html#data-serialization. Default: ${defaultParams.kryo}")
        .action((x, c) => c.copy(kryo = x))
      opt[String](FORMAT_TYPE_OPTION)
        .text(s"Input Avro file's format, which contains the information of each field's name. " +
        s"Options: [${FieldNamesType.values.mkString("|")}]. Default: ${defaultParams.fieldsNameType}")
        .action((x, c) => c.copy(fieldsNameType = FieldNamesType.withName(x.toUpperCase)))
      opt[Int](MIN_NUM_PARTITIONS_OPTION)
        .text(s"The minimum number of Hadoop splits to generate.. Default: ${defaultParams.minNumPartitions}. " +
        s"This would be potentially helpful when the number of default Hadoop splits is small. Note that when " +
        s"the default number of Hadoop splits (from HDFS) is larger than min-partitions, then min-partitions " +
        s"will be ignored and the number of partitions of the resulting RDD will be same as the default number " +
        s"of Hadoop splits. In short, min-partitions will *only* be able to increase the number of partitions.")
        .action((x, c) => c.copy(minNumPartitions = x))
      opt[Boolean](VALIDATE_PER_ITERATION)
        .text(s"If validating data is provided, and optimization tracker is enabled, " +
        s"whether to compute the evaluation metrics on validating data per iteration")
        .action((x, c) => c.copy(validatePerIteration = x))
      opt[String](SUMMARIZATION_OUTPUT_DIR)
        .text("An optional directory for summarization output")
        .action((x, c) => c.copy(summarizationOutputDirOpt = Some(x)))
      opt[String](NORMALIZATION_TYPE)
        .text(s"The normalization type to use in the training. Options: [${NormalizationType.values().mkString("|")}]. Default: ${defaultParams.normalizationType}")
        .action((x, c) => c.copy(normalizationType = NormalizationType.valueOf(x.toUpperCase)))
      opt[String](COEFFICIENT_BOX_CONSTRAINTS)
          .text(s"JSON array of maps specifying bound constraints on coefficients. The input is expected to be an array " +
          s"of maps with the map containing keys only from {" + ConstraintMapKeys.values.mkString(s",") + "} (other " +
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
                         })
          .action((x, c) => c.copy(constraintString = Some(x)))
      help(HELP_OPTION).text("prints ML-Ease's usage text")
      override def showUsageOnError = true
      checkConfig { c =>
        if ((c.regularizationType == RegularizationType.L1 || c.regularizationType == RegularizationType.ELASTIC_NET) && c.optimizerType == OptimizerType.TRON) {
          failure(s"Combination of (${c.regularizationType}, ${c.optimizerType}) is not allowed")
        } else if (!c.constraintString.isEmpty && c.normalizationType != NormalizationType.NO_SCALING) {
          failure(s"Normalization and box constraints should not be used together since we cannot guarantee the " +
              s"satisfaction of the coefficient constraints after normalization")
        } else {
          success
        }
      }
    }
    parser.parse(args, Params()) match {
      case Some(params) => postProcess(params)
      case None => throw new IllegalArgumentException(s"Parsing the command line arguments failed.\n" +
        s"Input arguments are: ${args.mkString(", ")}).")
    }
  }

  /**
   * This method post process params for necessary value override.
   * @param inputParams Parsed params
   * @return Post processed params
   */
  private def postProcess(inputParams: Params): Params = {
    if (inputParams.regularizationType == RegularizationType.NONE) {
      inputParams.copy(regularizationWeights = List(0.0))
    } else {
      inputParams
    }
  }
}
