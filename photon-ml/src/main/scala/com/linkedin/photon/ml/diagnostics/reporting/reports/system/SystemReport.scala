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
package com.linkedin.photon.ml.diagnostics.reporting.reports.system

import com.linkedin.photon.ml.Params
import com.linkedin.photon.ml.diagnostics.reporting.LogicalReport
import com.linkedin.photon.ml.stat.BasicStatisticalSummary

/**
 * Composite containing all of the system-related (i.e. common to all models) information.
 *
 * @param nameToIndex
 *                    Mapping of encoded (name, term) tuples &rarr; index
 * @param params
 *               Parameters used to launch the driver
 *
 * @param summary
 *                Computed feature summary
 */
case class SystemReport(val nameToIndex: Map[String, Int], var params: Params, var summary: Option[BasicStatisticalSummary] = None) extends LogicalReport
