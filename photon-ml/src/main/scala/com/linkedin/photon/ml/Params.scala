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


import com.linkedin.photon.ml.DataValidationType._
import com.linkedin.photon.ml.io.FieldNamesType._
import com.linkedin.photon.ml.normalization.NormalizationType
import com.linkedin.photon.ml.optimization.OptimizerType._
import com.linkedin.photon.ml.optimization.{OptimizerType, RegularizationType}
import com.linkedin.photon.ml.optimization.RegularizationType._
import com.linkedin.photon.ml.supervised.TaskType._

import scala.collection.mutable.ArrayBuffer


/**
 *  A bean class for PhotonML parameters to replace the original case class for input parameters.
 *
 * @author xazhang
 * @author dpeng
 * @author nkatariy
 */
class Params {
  /**
   * Training data directory
   */
  var trainDir: String = _
  /**
   * Validating data directory. Note that
   *    1. Validation is optional
   *    2. If validation data set is provided, the best model will be provided;
   *    3. No matter 1 or 2, all learned models will be provided, so users can always perform an independent model
   *       selection job
   */
  var validateDirOpt: Option[String] = None
  /**
   * Photon-ML's output directory
   */
  var outputDir: String = _
  /**
   * Learning task type, e.g., LINEAR_REGRESSION, POISSON_REGRESSION or LOGISTIC_REGRESSION
   */
  var taskType: TaskType = _
  /**
   * Maximum number of iterations to run
   */
  var maxNumIter: Int = 80
  /**
   * An array of regularization weights that will be used to train the model
   */
  var regularizationWeights: List[Double] = List(10)
  /**
   * The optimizer's convergence tolerance, smaller value will lead to higher accuracy with the cost of more iterations
   */
  var tolerance: Double = 1e-6
  /**
   * The type of optimizer that will be used to train the model
   */
  var optimizerType: OptimizerType = LBFGS
  /**
   * The type of regularization that will be used to train the model
   */
  var regularizationType: RegularizationType = L2
  var elasticNetAlpha: Option[Double] = None
  /**
   * Whether to learn the intercept
   */
  var addIntercept: Boolean = true
  /**
   * Whether to enable the optimization tracker, which stores the per-iteration log information of the running optimizer
   */
  var enableOptimizationStateTracker: Boolean = true
  /**
   * If validating data is provided, and optimization tracker is enabled,
   * whether to compute the evaluation metrics on validating data per iteration
   */
  var validatePerIteration: Boolean = false
  /**
   * The minimum number of Hadoop splits to generate. This would be potentially helpful when the
   * number of default Hadoop splits is small. Note that when the default number of Hadoop splits
   * (from HDFS) is larger than minNumPartitions, then minNumPartitions will be ignored and the
   * number of partitions of the resulting RDD will be same as the default number of Hadoop splits.
   * In short, minNumPartitions will *only* be able to increase the number of partitions.
   */
  var minNumPartitions: Int = 1
  /**
   * Whether to use kryo to serialize the intermediate result, which is generally a good idea. More information
   * on serialization method used in Spark can be found through the following link:
   * [[https://spark.apache.org/docs/latest/tuning.html#data-serialization]].
   */
  var kryo: Boolean = true
  /**
   * Input Avro file's format, which contains the information of each field's name
   */
  var fieldsNameType: FieldNamesType = RESPONSE_PREDICTION
  /**
   * If summarization output dir is provided, basic statistics of features will be written to the given directory.
   */
  var summarizationOutputDirOpt: Option[String] = None
  /**
   * Feature normalization method
   */
  var normalizationType: NormalizationType = NormalizationType.NONE
  /**
   * Job name of this spark application
   */
  var jobName: String = s"Photon-ML-Training"
  /**
   * Type of validation to be performed
   */
  var dataValidationType: DataValidationType = DataValidationType.VALIDATE_FULL
  /**
   * A JSON string containing an array of maps specifying the box constraints on certain
   * coefficients, if any. Only keys from
   * [[com.linkedin.photon.ml.io.ConstraintMapKeys]] will be sought. Others if
   * specified will be ignored. The term is allowed to be a wildcard "*" in which case the bounds
   * are applied to all features with the specified name irrespective of the term. The name cannot
   * be a wildcard except for the special case where both name and term are wildcards so that one
   * wants to apply the same bound to all coefficients
   */
  var constraintString: Option[String] = None
  /**
   * Control whether training diagnostics like bootstrapping, etc. should be run. Since these
   * tend to be more computationally demanding, this should default to false.
   */
  var trainingDiagnosticsEnabled: Boolean = false
  /**
   * A file containing selected features. The file is expected to contain avro records that have
   * the "name" and "term" fields
   */
  var selectedFeaturesFile: Option[String] = None
  /**
   * The depth used in tree aggregate
   */
  var treeAggregateDepth: Int = 1

  var offHeapIndexMapDir: Option[String] = None
  var offHeapIndexMapNumPartitions: Int = 0

  /**
   * Validate this parameters. Exception will be thrown if the parameter combination is invalid.
   */
  def validate(): Unit = {
    val messages = new ArrayBuffer[String]
    if ((regularizationType == RegularizationType.L1 ||
            regularizationType == RegularizationType.ELASTIC_NET) && optimizerType == OptimizerType.TRON) {
      messages += s"Combination of ($regularizationType, $optimizerType) is not allowed."
    }
    if (constraintString.nonEmpty && normalizationType != NormalizationType.NONE) {
      messages += (s"Normalization and box constraints should not be used together since we cannot guarantee the " +
                   s"satisfaction of the coefficient constraints after normalization.")
    }
    if (normalizationType == NormalizationType.STANDARDIZATION && !addIntercept) {
      messages += s"Intercept must be used to enable feature standardization. Normalization type: " +
                  s"$normalizationType, add intercept: $addIntercept."
    }
    if (messages.nonEmpty) {
      throw new IllegalArgumentException(messages.mkString("\n"))
    }
  }

  override def toString: String = {
    getClass.getDeclaredFields.map(field => {
      field.setAccessible(true)
      field.getName + "=" + field.get(this)
    }).mkString("\n")
  }
}
