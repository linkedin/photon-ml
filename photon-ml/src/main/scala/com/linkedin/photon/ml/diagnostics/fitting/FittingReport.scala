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
package com.linkedin.photon.ml.diagnostics.fitting

import com.linkedin.photon.ml.diagnostics.reporting.LogicalReport

/**
 * Demonstrate how model metrics change as a function of the volume of data used to fit the model, both on the training
 * set and a held-out set
 *
 * @param metrics
 *                Map of (metric name &rarr; (% training set, performance on training set, performance on held out) tuples
 *
 * @param fittingMsg
 *                   Description of any questions / comments / concerns that came up while testing how well we fit
 */
case class FittingReport(val metrics:Map[String, (Array[Double], Array[Double], Array[Double])],
                         val fittingMsg:String) extends LogicalReport
