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

import com.linkedin.photon.ml.util.MathUtils

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

  /**
   * Operation used to compare two scores. In some cases, the better score is higher (e.g. AUC) and in others, the
   * better score is lower (e.g. RMSE).
   */
  val op: (Double, Double) => Boolean

  /**
   * Determine the better between two scores for this evaluation metric.
   *
   * @param score1 The first score
   * @param score2 The second score
   * @return True if the first score is better than the second score, false otherwise
   */
  def betterThan(score1: Double, score2: Double): Boolean = op(score1, score2)

  /**
   * Returns a string representation of the [[EvaluatorType]]
   *
   * @return The name of the [[EvaluatorType]]
   */
  override def toString: String = name
}

object EvaluatorType {

  // Comparable to the valueSet, if this were an enumeration
  val all: Seq[EvaluatorType] = Seq(AUC, AUPR, RMSE, LogisticLoss, PoissonLoss, SquaredLoss)

  case object AUC extends EvaluatorType { val name = "AUC"; val op = MathUtils.greaterThan _ }
  case object AUPR extends EvaluatorType { val name = "AUPR"; val op = MathUtils.greaterThan _ }
  case object RMSE extends EvaluatorType { val name = "RMSE"; val op = MathUtils.lessThan _ }
  case object LogisticLoss extends EvaluatorType { val name = "LOGISTIC_LOSS"; val op = MathUtils.lessThan _ }
  case object PoissonLoss extends EvaluatorType { val name = "POISSON_LOSS"; val op = MathUtils.lessThan _ }
  case object SquaredLoss extends EvaluatorType { val name = "SQUARED_LOSS"; val op = MathUtils.lessThan _ }
}
