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
package com.linkedin.photon.ml.io

/**
 * Enum of the keys that are expected to be present in the maps in the user-specified constraint string
 * The "name" and "term" keys are mandatory while one of "lowerBound" and "upperBound" is allowed to be skipped in
 * which case it is assumed to be -Inf / +Inf appropriately
 *
 * @author nkatariy
 */
object ConstraintMapKeys extends Enumeration {
  type ConstraintMapKeys = Value
  val name, term, lowerBound, upperBound = Value
}
