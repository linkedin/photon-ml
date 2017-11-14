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
package com.linkedin.photon.ml.evaluation

/**
 * Evaluator type
 */
trait EvaluatorType {

  /**
   * Name of the evaluator.
   *
   * @note It is currently also used as the cli input argument for evaluator types in integTests.
   */
  val name: String

  override def toString: String = name
}

object EvaluatorType {

  // Comparable to the valueSet, if this were an enumeration
  val all: Seq[EvaluatorType] = Seq(AUC, RMSE, LogisticLoss, PoissonLoss, SmoothedHingeLoss, SquaredLoss)

  case object AUC extends EvaluatorType { val name = "AUC" }
  case object RMSE extends EvaluatorType { val name = "RMSE" }
  case object LogisticLoss extends EvaluatorType { val name = "LOGISTIC_LOSS" }
  case object PoissonLoss extends EvaluatorType { val name = "POISSON_LOSS" }
  case object SmoothedHingeLoss extends EvaluatorType { val name = "SMOOTHED_HINGE_LOSS" }
  case object SquaredLoss extends EvaluatorType { val name = "SQUARED_LOSS" }
}
