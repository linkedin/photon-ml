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
package com.linkedin.photon.ml.optimization.game

/**
 * Configuration object for matrix factorization optimization.
 *
 * @param maxNumberIterations Maximum number of iterations
 * @param numFactors Number of factors
 */
protected[ml] case class MFOptimizationConfiguration(maxNumberIterations: Int, numFactors: Int) {
  override def toString: String = {
    s"maxNumberIterations: $maxNumberIterations\tnumFactors: $numFactors"
  }
}

object MFOptimizationConfiguration {

  protected[ml] val SPLITTER = ","
  protected[ml] val EXPECTED_FORMAT: String = s"maxNumberIterations${SPLITTER}numFactors{SPLITTER}"
  protected[ml] val EXPECTED_NUM_CONFIGS = 2

  /**
   * Parse and build the configuration object from a string representation.
   *
   * TODO: Add assert and meaningful parsing error message here
   *
   * @param string The string representation
   */
  protected[ml] def parseAndBuildFromString(string: String): MFOptimizationConfiguration = {
    val configParams = string.split(SPLITTER).map(_.trim)
    require(configParams.length == EXPECTED_NUM_CONFIGS,
      s"Parsing $string failed! The expected MF optimization configuration should contain $EXPECTED_NUM_CONFIGS " +
          s"parts separated by \'$SPLITTER\', but found ${configParams.length}. Expected format: $EXPECTED_FORMAT")

    val Array(maxNumberIterations, numFactors) = configParams.map(_.toInt)
    MFOptimizationConfiguration(maxNumberIterations, numFactors)
  }
}
