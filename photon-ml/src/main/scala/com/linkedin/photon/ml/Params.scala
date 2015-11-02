package com.linkedin.photon.ml

import com.linkedin.photon.ml.DataValidationType.DataValidationType
import com.linkedin.photon.ml.io.FieldNamesType
import FieldNamesType._
import com.linkedin.photon.ml.optimization.{RegularizationType, OptimizerType}
import OptimizerType._
import RegularizationType._
import com.linkedin.photon.ml.supervised.TaskType
import TaskType._
import com.linkedin.photon.ml.normalization.NormalizationType

/**
 * Photon-ML's parameters
 * @param trainDir Training data directory
 * @param validateDirOpt Validating data directory. Note that
    1. Validation is optional
    2. If validation data set is provided, the best model will be provided;
    3. No matter 1 or 2, all learned models will be provided, so users can always perform an independent model selection job
 * @param outputDir Photon-ML's output directory
 * @param taskType Learning task type, e.g., LINEAR_REGRESSION, POISSON_REGRESSION or LOGISTIC_REGRESSION
 * @param addIntercept Whether to learn the intercept
 * @param regularizationWeights An array of regularization weights that will be used to train the model
 * @param regularizationType The type of regularization that will be used to train the model
 * @param optimizerType The type of optimizer that will be used to train the model
 * @param maxNumIter Maximum number of iterations to run
 * @param tolerance The optimizer's convergence tolerance, smaller value will lead to higher accuracy with the cost of more iterations
 * @param enableOptimizationStateTracker Whether to enable the optimization tracker, which stores the per-iteration log information of the running optimizer
 * @param validatePerIteration If validating data is provided, and optimization tracker is enabled,
 *                             whether to compute the evaluation metrics on validating data per iteration
 * @param minNumPartitions The minimum number of Hadoop splits to generate. This would be potentially helpful when the
 *                         number of default Hadoop splits is small. Note that when the default number of Hadoop splits
 *                         (from HDFS) is larger than minNumPartitions, then minNumPartitions will be ignored and the
 *                         number of partitions of the resulting RDD will be same as the default number of Hadoop splits.
 *                         In short, minNumPartitions will *only* be able to increase the number of partitions.
 * @param kryo Whether to use kryo to serialize the intermediate result, which is generally a good idea. More information
 *             on serialization method used in Spark can be found through the following link:
 *             [[https://spark.apache.org/docs/latest/tuning.html#data-serialization]].
 * @param fieldsNameType Input Avro file's format, which contains the information of each field's name
 * @param summarizationOutputDirOpt If summarization output dir is provided, basic statistics of features will be written to the given directory.
 * @param normalizationType Feature normalization method
 * @param jobName Job name of this spark application
 * @param dataValidationType Type of validation to be performed
 * @param constraintString A JSON string containing an array of maps specifying the box constraints on certain
 *                         coefficients, if any. Only keys from
 *                         {@see com.linkedin.photon.ml.io.GLMSuite.ConstraintMapKeys} will be sought. Others if
 *                         specified will be ignored. The term is allowed to be a wildcard "*" in which case the bounds
 *                         are applied to all features with the specified name irrespective of the term. The name cannot
 *                         be a wildcard except for the special case where both name and term are wildcards so that one
 *                         wants to apply the same bound to all coefficients
 * @author xazhang
 * @author dpeng
 */
case class Params(trainDir: String = null,
                  validateDirOpt: Option[String] = None,
                  outputDir: String = null,
                  taskType: TaskType = null,
                  maxNumIter: Int = 80,
                  regularizationWeights: List[Double] = List(0.1, 1, 10, 100),
                  tolerance: Double = 1e-6,
                  optimizerType: OptimizerType = LBFGS,
                  regularizationType: RegularizationType = L2,
                  elasticNetAlpha: Option[Double] = None,
                  addIntercept: Boolean = true,
                  enableOptimizationStateTracker: Boolean = true,
                  validatePerIteration: Boolean = false,
                  minNumPartitions: Int = 1,
                  kryo: Boolean = true,
                  fieldsNameType: FieldNamesType = RESPONSE_PREDICTION,
                  summarizationOutputDirOpt: Option[String] = None,
                  normalizationType: NormalizationType = NormalizationType.NONE,
                  jobName: String = s"Photon-ML-Training",
                  dataValidationType: DataValidationType = DataValidationType.VALIDATE_FULL,
                  constraintString: Option[String] = None)
