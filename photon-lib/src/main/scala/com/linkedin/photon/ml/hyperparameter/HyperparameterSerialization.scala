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

import breeze.linalg.DenseVector
import com.linkedin.photon.ml.HyperparameterTuningMode
import com.linkedin.photon.ml.util.DoubleRange

import scala.util.parsing.json.JSON


/**
 * An object to deserialize configuration and prior observations of hyper-parameters.
 */
object HyperparameterSerialization {

  val BAYESIAN_MODEL = "BAYESIAN"
  val RANDOM_MODEL = "RANDOM"
  val LOG_TRANSFORM = "LOG"
  val SQRT_TRANSFORM = "SQRT"

  /**
   * Parse a [[Seq]] of prior observations for hyper-parameter tuning from JSON format.
   *
   * @param priorDataJson The JSON containing prior observations
   * @param priorDefault Default values for missing hyper-parameters
   * @param hyperParameterList The list of hyper-parameters to tune
   * @return A [[Seq]] of (vectorized hyper-parameter settings, evaluationValue) tuples
   */
  def priorFromJson(
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

  /**
   * Read in the JSON config file and returns the parsed hyper-parameter configurations.
   *
   * @param jsonConfig The config file in json format.
   * @return A tuple containing the hyperparameter tuning mode, list of parameters, ranges of min and max for each
   *         parameter.
   */
  def configFromJson(jsonConfig: String): HyperparameterConfig = {

    val (tuningMode, paramDetails) = JSON.parseFull(jsonConfig) match {
      case Some(inputConfig: Map[String, Any]) =>

        val mode = inputConfig("tuning_mode") match {
          case BAYESIAN_MODEL => HyperparameterTuningMode.BAYESIAN
          case RANDOM_MODEL => HyperparameterTuningMode.RANDOM
          case _ => HyperparameterTuningMode.NONE
        }

        val hyperparameterDetails = inputConfig("variables") match {

          case variables: Map[String, Any] =>

            variables.map {
              case (key: String, value: Map[String, Any]) =>

                (value("type"), value("min"), value("max"), value.get("transform")) match {
                  case (varType: String, min: Double, max: Double, transform: Option[_]) =>
                    (key, varType, min, max, transform)

                  case _ => throw new IllegalArgumentException("The minimum and maximum values must be numeric")
                }

              case _ =>
                throw new IllegalArgumentException("Each hyper-parameter configuration must be a map")
            }

          case _ =>
            throw new IllegalArgumentException("The hyper-parameter configurations must be a map")
        }

        (mode, hyperparameterDetails)

      case _ =>
        throw new IllegalArgumentException("JSON config is not a Map[String, Any]")
    }

    val hyperparameters = paramDetails.map(_._1).toSeq
    val discreteParams = paramDetails.zipWithIndex.filter(_._1._2 == "INT").map {
      case ((_, _, min: Double, max: Double, _), index: Int) => index -> ((max - min).toInt + 1)
    }.toMap
    val ranges = paramDetails.map { case (_, _, min, max, _) => DoubleRange(min, max) }.toSeq

    val transformMap = paramDetails.map(_._5).zipWithIndex.flatMap {
      case (transform, index) => transform.map(trans => trans.toString match {
        case LOG_TRANSFORM | SQRT_TRANSFORM => index -> trans.toString
        case _ => throw new IllegalArgumentException("The transformation is not valid")
      })
    }.toMap

    HyperparameterConfig(tuningMode, hyperparameters, ranges, discreteParams, transformMap)
  }
}
