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
package com.linkedin.photon.ml

/**
 * A collection of option names used in Photon-ML.
 */
object PhotonOptionNames {

  val HELP_OPTION = "help"
  val TRAIN_DIR_OPTION = "training-data-directory"
  val VALIDATE_DIR_OPTION = "validating-data-directory"
  val OUTPUT_DIR_OPTION = "output-directory"
  val JOB_NAME_OPTION = "job-name"
  val REGULARIZATION_WEIGHTS_OPTION = "regularization-weights"
  val INTERCEPT_OPTION = "intercept"
  val MAX_NUM_ITERATIONS_OPTION = "num-iterations"
  val TOLERANCE_OPTION = "convergence-tolerance"
  val TASK_TYPE_OPTION = "task"
  val FORMAT_TYPE_OPTION = "format"
  val OPTIMIZER_TYPE_OPTION = "optimizer"
  val REGULARIZATION_TYPE_OPTION = "regularization-type"
  val ELASTIC_NET_ALPHA_OPTION = "elastic-net-alpha"
  val KRYO_OPTION = "kryo"
  val OPTIMIZATION_STATE_TRACKER_OPTION = "optimization-tracker"
  val MIN_NUM_PARTITIONS_OPTION = "min-partitions"
  val VALIDATE_PER_ITERATION = "validate-per-iteration"
  val SUMMARIZATION_OUTPUT_DIR = "summarization-output-dir"
  val NORMALIZATION_TYPE = "normalization-type"
  val COEFFICIENT_BOX_CONSTRAINTS = "coefficient-box-constraints"
  val DATA_VALIDATION_TYPE = "data-validation-type"
  val TREE_AGGREGATE_DEPTH = "tree-aggregate-depth"
  val DIAGNOSTIC_MODE = "diagnostic-mode"
  val SELECTED_FEATURES_FILE = "selected-features-file"
  val OFFHEAP_INDEXMAP_DIR = "offheap-indexmap-dir"
  val OFFHEAP_INDEXMAP_NUM_PARTITIONS = "offheap-indexmap-num-partitions"
  val DELETE_OUTPUT_DIRS_IF_EXIST = "delete-output-dirs-if-exist"
  val EVENT_LISTENERS = "event-listeners"
  val FEATURE_DIMENSION = "feature-dimension"
  val INPUT_FILE_FORMAT = "input-file-format"
  val USE_WARM_START = "use-warm-start"

  @deprecated("Use 'diagnostic-mode'", "1.0")
  val TRAINING_DIAGNOSTICS = "training-diagnostics"
}
