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
package com.linkedin.photon.ml.event

import org.apache.spark.SparkContext
import org.slf4j.Logger

import com.linkedin.photon.ml.Evaluation.MetricsMap
import com.linkedin.photon.ml.Params
import com.linkedin.photon.ml.supervised.model.ModelTracker

/**
 * Base trait for all consumable events.
 */
abstract class Event

/**
 * Photon model training setup event.
 *
 * @param logger The custom logger for the Spark job
 * @param sc The SparkContext for the Spark job
 * @param params The user-defined parameters for the Photon job
 */
case class PhotonSetupEvent(logger: Logger, sc: SparkContext, params: Params) extends Event

/**
 * Photon training start event.
 *
 * @param time The Unix time at which training began
 */
case class TrainingStartEvent(time: Long) extends Event

/**
 * Photon training end event.
 *
 * @param time The Unix time at which training concluded
 */
case class TrainingFinishEvent(time: Long) extends Event

/**
 * Event containing log information for the full optimization of a single Photon model.
 *
 * @param regWeight The regularization weight of the trained model
 * @param modelTracker The optimization states and intermediate models
 * @param perIterationMetrics The validation metrics for the intermediate models during optimization
 * @param finalMetrics The validation metrics for the trained model
 */
case class PhotonOptimizationLogEvent(
    regWeight: Double,
    modelTracker: ModelTracker,
    perIterationMetrics: Option[Array[MetricsMap]] = None,
    finalMetrics: Option[MetricsMap] = None) extends Event
