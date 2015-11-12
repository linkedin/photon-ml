# Copyright 2014 LinkedIn Corp. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain a
# copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.
#!/bin/bash
# #####################
# This is a script that demonstrates how to run photon ml from command line.
# It is only recommended to be used for dev purpose and please be aware a different user/app potentially need to adjust
# almost all the parameters to make the job running.
#
# ######################

JOB_NAME="run-photon-ml-driver"

WORKING_ROOT_DIR="/user/yizhou/run_photon_ml_driver"
TRAIN_ROOT="${WORKING_ROOT_DIR}/input"
TRAIN_INPUT_DIRS="${TRAIN_ROOT}/train"
VALIDATE_INPUT_DIRS="${TRAIN_ROOT}/test"
SUMMARY_OUTPUT_PATH="${WORKING_ROOT_DIR}/summary"
OUTPUT_DIR="${WORKING_ROOT_DIR}/results"

PHOTON_MAIN_JAR=photon-ml_2.10-1.1.7-SNAPSHOT.jar
LIBJARS="joda-time-2.7.jar,avro-1.7.5.jar,avro-mapred-1.7.5.jar,scopt_2.10-3.2.0.jar,photon-schemas-0.0.2.jar,xchart-2.5.1.jar,batik-util-1.7.jar,batik-awt-util-1.7.jar,batik-svggen-1.7.jar"
ADMIN_ACLS="$USER"
QUEUE="default"

EXECUTOR_MEMORY_MB=$((2 * 1024))
EXECUTOR_MEMORY="${EXECUTOR_MEMORY_MB}m" # amount of memory requested per container
EXECUTOR_MEMORY_OVERHEAD=$((EXECUTOR_MEMORY_MB / 4))

NUM_EXECUTORS=10 # number of container/executor requested for this application
MIN_PARTITIONS=$((NUM_EXECUTORS * 3))

DRIVER_MEMORY_MB=$((5 * 1024))
DRIVER_MEMORY="${DRIVER_MEMORY_MB}m" # amount of memory requested for driver
AKKA_FRAME_SIZE=256 # in MBs
KRYO_BUFFER_MAX="$((2 * 1024 - 1))m" # kryo.buffer.max must be < 2048m

MAX_RESULT_SIZE_MB=$((DRIVER_MEMORY_MB / 4))
MAX_RESULT_SIZE="${MAX_RESULT_SIZE_MB}m"

TASK_TYPE="LOGISTIC_REGRESSION"
INPUT_FORMAT="TRAINING_EXAMPLE"
NUM_ITERATIONS=20
LAMBDAS="1,10,50"
TOLERANCE="1e-5"
OPTMIZER_TYPE="TRON"
REGULARIZATION_TYPE="L2"
NORMALIZATION_TYPE="NONE"
VALIDATE_PER_ITERATION=true
USE_OPTIMIZATION_TRACKER=true
USE_INTERCEPT=true
# UNCOMMENT to use different options:
# ELASTIC_NET_ALPHA=0.1
# BOX_CONSTRAINTS=

if [ ! -z $ELASTIC_NET_ALPHA ]
then
  REGULARIZATION_TYPE_ARG="--regularization-type ELASTIC_NET"
  ELASTIC_NET_ARG="--elastic-net-alpha $ELASTIC_NET_ALPHA"
else
  REGULARIZATION_ARG="--regularization-type $REGULARIZATION_TYPE"
fi

if [ ! -z $BOX_CONSTRAINTS ]
then
  BOX_CONSTRAINTS_ARG="--coefficients-box-constraints $BOX_CONSTRAINTS"
fi

CMD_TO_RUN="
spark-submit \
--class com.linkedin.photon.ml.Driver \
--queue $QUEUE \
--master yarn-cluster \
--num-executors $NUM_EXECUTORS \
--driver-memory $DRIVER_MEMORY \
--executor-memory $EXECUTOR_MEMORY \
--conf \"spark.executor.extraJavaOptions=-XX:+PrintGCDetails -XX:+PrintGCTimeStamps\" \
--conf spark.ui.showConsoleProgress=false \
--conf spark.eventLog.enabled=true \
--conf spark.admin.acls=$ADMIN_ACLS \
--conf spark.driver.maxResultSize=$MAX_RESULT_SIZE \
--conf spark.shuffle.consolidateFiles=true \
--conf spark.yarn.executor.memoryOverhead=$EXECUTOR_MEMORY_OVERHEAD \
--conf spark.shuffle.io.preferDirectBufs=false \
--conf spark.akka.frameSize=$AKKA_FRAME_SIZE \
--conf spark.kryoserializer.buffer.max=$KRYO_BUFFER_MAX \
--jars $LIBJARS  \
$PHOTON_MAIN_JAR \
--job-name $JOB_NAME \
--task $TASK_TYPE \
--training-data-directory $TRAIN_INPUT_DIRS \
--output-directory $OUTPUT_DIR \
--validate-per-iteration $VALIDATE_PER_ITERATION \
--validating-data-directory $VALIDATE_INPUT_DIRS \
--optimization-tracker $USE_OPTIMIZATION_TRACKER \
--intercept $USE_INTERCEPT \
--convergence-tolerance $TOLERANCE \
--format $INPUT_FORMAT \
--optimizer $OPTMIZER_TYPE \
--normalization-type $NORMALIZATION_TYPE \
--num-iterations $NUM_ITERATIONS \
--summarization-output-dir $SUMMARY_OUTPUT_PATH \
--regularization-weights $LAMBDAS \
--min-partitions $MIN_PARTITIONS \
$REGULARIZATION_TYPE_ARG \
$ELASTIC_NET_ARG \
$BOX_CONSTRAINTS_ARG
"
echo "Execute command: $CMD_TO_RUN"

eval $CMD_TO_RUN
