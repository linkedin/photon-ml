/*
 * Copyright 2018 LinkedIn Corp. All rights reserved.
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
package com.linkedin.photon.ml.util

import org.apache.commons.cli.MissingArgumentException
import org.apache.spark.ml.param.{Param, ParamMap, Params}

/**
 * Extension of Spark [[Params]] class for Photon ML.
 */
trait PhotonParams extends Params {

  /**
   * Return the user-supplied value for a required parameter. Used for mandatory parameters without default values.
   *
   * @tparam T The type of the parameter
   * @param param The parameter
   * @return The value associated with the parameter
   * @throws MissingArgumentException if no value is associated with the given parameter
   */
  protected def getRequiredParam[T](param: Param[T]): T =
    get(param).getOrElse(throw new MissingArgumentException(s"Missing required parameter ${param.name}"))

  /**
   * Set the default parameters.
   */
  protected def setDefaultParams(): Unit

  /**
   * Check that all required parameters have been set and validate interactions between parameters.
   *
   * @note In Spark, interactions between parameters are checked by
   *       [[org.apache.spark.ml.PipelineStage.transformSchema()]]. Since we do not use the Spark pipeline API in
   *       Photon-ML, we need to have this function to check the interactions between parameters.
   * @throws MissingArgumentException if a required parameter is missing
   * @throws IllegalArgumentException if a validation check fails
   * @param paramMap The parameters to validate
   */
  def validateParams(paramMap: ParamMap): Unit

  /**
   * Clear all set parameters.
   */
  def clear(): Unit = params.foreach(clear)
}
