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
package com.linkedin.photon.ml.optimization.game

/**
 * Configuration object for matrix factorization optimization
 *
 * @param maxNumberIterations maximum number of iterations
 * @param numFactors number of factors
 * @author xazhang
 */
case class MFOptimizationConfiguration(maxNumberIterations: Int, numFactors: Int) {
  override def toString: String = {
    s"maxNumberIterations: $maxNumberIterations\tnumFactors: $numFactors"
  }
}

object MFOptimizationConfiguration {

  /**
   * Parse and build the configuration object from a string representation
   *
   * @param string the string representation
   * @todo Add assert and meaningful parsing error message here
   */
  def parseAndBuildFromString(string: String): MFOptimizationConfiguration = {
    val Array(maxNumberIterations, numFactors) = string.split(",").map(_.toInt)
    MFOptimizationConfiguration(maxNumberIterations, numFactors)
  }
}
