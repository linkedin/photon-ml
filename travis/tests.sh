#!/usr/bin/env bash
#
# This script is used to customize Travis CI testing as follows:
# - Wrapper for gradle call to run unit and integration tests in parallel
# - Redirect integTest output to tmp file to avoid Travis CI 4MB log file limit. In addition, ping Travis CI while
#   testing to avoid the 10 minute no-output timeout. This code modified from the base script found here:
#   http://stackoverflow.com/questions/26082444/how-to-work-around-travis-cis-4mb-output-limit
#

NUM_LINES=4000

# Helper functions
dump_output() {
  echo "Dumping the last $NUM_LINES lines of output:"
  tail -${NUM_LINES} ${BUILD_OUTPUT}
}

kill_ping_loop() {
  kill ${PING_LOOP_PID}
}

clean_up() {
  # The build finished without returning an error so dump a tail of the output
  dump_output
  # Nicely terminate the ping output loop
  kill_ping_loop
}

error_handler() {
  echo "ERROR: An error was encountered with the build."
  clean_up
  exit 1
}


# Abort on error
set -e

# Clean up any previous build artifacts
./gradlew clean

if [[ $# -ne 1 ]]; then
  echo "ERROR: Wrong # of arguments"
  exit 1
fi

SCALA_SUFFIX="${TRAVIS_SCALA_VERSION%.[0-9]}"

if [[ "$1" == "integration" ]]; then
  # Redirect output to file, ping Travis intermittently so it doesn't kill the test.
  PING_SLEEP=30s
  BUILD_OUTPUT=/tmp/build.out

  touch ${BUILD_OUTPUT}

  # If an error occurs, run our error handler to output a tail of the build
  trap 'error_handler' ERR

  # Set up a repeating loop to send some output to Travis.
  bash -c "while true; do echo \"\$(date) - testing ...\"; sleep $PING_SLEEP; done" &
  PING_LOOP_PID=$!

  # Run integration tests, redirect output to tmp file
  ./gradlew integTest -Pexclude=ml/DriverTest,deprecated/GLMSuiteIntegTest &> ${BUILD_OUTPUT} -Pv=${SCALA_SUFFIX}

  # Kill ping process, output logs
  clean_up

elif [[ "$1" == "unit" ]]; then
  ./gradlew test -Pv=${SCALA_SUFFIX}

elif [[ "$1" == "rat" ]]; then
  ./gradlew rat

else
  echo "ERROR: Invalid test type specified; must be either 'unit' or 'integration'"
  exit 1
fi
