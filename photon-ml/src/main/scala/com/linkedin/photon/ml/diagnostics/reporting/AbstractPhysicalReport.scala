/*
 * Copyright 2014 LinkedIn Corp. All rights reserved.
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
package com.linkedin.photon.ml.diagnostics.reporting

import java.util.concurrent.atomic.AtomicLong

/**
 * Created by bdrew on 10/9/15.
 */
class AbstractPhysicalReport extends PhysicalReport {
  private val id:Long = AbstractPhysicalReport.GLOBAL_ID_COUNTER.getAndIncrement()
  def getId():Long = id
}

object AbstractPhysicalReport {
  val GLOBAL_ID_COUNTER:AtomicLong = new AtomicLong(0L)
}