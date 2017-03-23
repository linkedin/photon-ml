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
  * Represents the complete Game optimization configuration as it appeared for a single run with a particular setting
  * for each hyperparameter.
  *
  * @param fixedEffectOptimizationConfiguration the optimization configuration for each fixed effect
  * @param randomEffectOptimizationConfiguration the optimization configuration for each random effect
  * @param factoredRandomEffectOptimizationConfiguration the optimization configuration for each factored random effect
  */
case class GameModelOptimizationConfiguration(
    fixedEffectOptimizationConfiguration: Map[String, GLMOptimizationConfiguration],
    randomEffectOptimizationConfiguration: Map[String, GLMOptimizationConfiguration],
    factoredRandomEffectOptimizationConfiguration: Map[String,
      (GLMOptimizationConfiguration, GLMOptimizationConfiguration, MFOptimizationConfiguration)]) {

  /**
    * Build a custom string representation of the configuration
    */
  override def toString() = Seq(fixedEffectOptimizationConfiguration.mkString("\n"),
    randomEffectOptimizationConfiguration.mkString("\n"),
    factoredRandomEffectOptimizationConfiguration.mkString("\n")).mkString("\n")
}
