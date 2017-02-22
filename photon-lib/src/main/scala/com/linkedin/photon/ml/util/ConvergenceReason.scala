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
package com.linkedin.photon.ml.util

/**
 * Represents reasons for convergence, with string descriptions (inspired by: breeze.optimize.FirstOrderMinimizer).
 */
sealed trait ConvergenceReason {
  def reason: String
}

case object MaxIterations extends ConvergenceReason {
  override def reason: String = "max iterations reached"
}

case object FunctionValuesConverged extends ConvergenceReason {
  override def reason: String = "function values converged"
}

case object GradientConverged extends ConvergenceReason {
  override def reason: String = "gradient converged"
}

case object ObjectiveNotImproving extends ConvergenceReason {
  override def reason: String = "objective is not improving"
}
