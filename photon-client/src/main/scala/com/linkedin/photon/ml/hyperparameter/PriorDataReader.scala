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
package com.linkedin.photon.ml.hyperparameter

import scala.util.parsing.json.JSON

import breeze.linalg.DenseVector

/**
 * Helper class to load prior observation data.
 */
object PriorDataReader {

  /**
   * Parse a [[Seq]] of prior observations for hyper-parameter tuning from JSON format.
   *
   * @param priorDataJson The JSON containing prior observations
   * @param priorDefault Default values for missing hyper-parameters
   * @param hyperParameterList The list of hyper-parameters to tune
   * @return A [[Seq]] of (vectorized hyper-parameter settings, evaluationValue) tuples
   */
  def fromJson(
      priorDataJson: String,
      priorDefault: Map[String, String],
      hyperParameterList: Seq[String]): Seq[(DenseVector[Double], Double)] = {

    val priorData = JSON.parseFull(priorDataJson) match {
      case Some(priorDataMap: Map[String, Any]) =>

        priorDataMap("records") match {
          case optionsList: Seq[Map[String, String]] =>

            optionsList.map { paramMap =>
              val evaluationValue = paramMap("evaluationValue").toDouble
              val sortedValues = hyperParameterList.map { paramName =>
                paramMap.getOrElse(paramName, priorDefault(paramName)).toDouble
              }

              (sortedValues, evaluationValue)
            }

          case _ =>
            throw new IllegalArgumentException("Each record is not a list of Map[String, String]")
        }

      case _ =>
        throw new IllegalArgumentException("The JSON file is not a Map")
    }

    priorData.map { case (params, evalValue) =>
      (DenseVector(params.toArray), evalValue)
    }
  }
}
