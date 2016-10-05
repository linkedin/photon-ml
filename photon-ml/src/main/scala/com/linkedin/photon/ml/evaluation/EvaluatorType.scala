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
package com.linkedin.photon.ml.evaluation

/**
 * Evaluator type
 */
trait EvaluatorType {
  /**
   * Name of the evaluator
   * @note It is currently also used as the cli input argument for evaluator types in integTests
   */
  val name: String
}

object AUC extends EvaluatorType {
  val name = "AUC"
}

object RMSE extends EvaluatorType {
  val name = "RMSE"
}

object LogisticLoss extends EvaluatorType {
  val name = "LOGISTIC_LOSS"
}

object PoissonLoss extends EvaluatorType {
  val name = "POISSON_LOSS"
}

object SmoothedHingeLoss extends EvaluatorType {
  val name = "SMOOTHED_HINGE_LOSS"
}

object SquaredLoss extends EvaluatorType {
  val name = "SQUARED_LOSS"
}

object EvaluatorType {

  // Command line argument for evaluator type
  val cmdArgument = "evaluator-type"

  /**
   * Parse the evaluator type with name
 *
   * @param name name of the evaluator type
   * @return the parsed evaluator type
   */
  def withName(name: String): EvaluatorType = name.trim.toUpperCase match {
    case AUC.name => AUC
    case RMSE.name => RMSE
    case LogisticLoss.name | "LOGISTICLOSS" => LogisticLoss
    case PoissonLoss.name | "POISSONLOSS" => PoissonLoss
    case SmoothedHingeLoss.name | "SMOOTHEDHINGELOSS" => SmoothedHingeLoss
    case SquaredLoss.name | "SQUAREDLOSS" => SquaredLoss
    case ShardedPrecisionAtK.shardedPrecisionAtKPattern(k, _) =>
      val ShardedPrecisionAtK.shardedPrecisionAtKPattern(_, idName) = name.trim
      ShardedPrecisionAtK(k.toInt, idName)
    case ShardedAUC.shardedAUCPattern(_) =>
      val ShardedAUC.shardedAUCPattern(idName) = name.trim
      ShardedAUC(idName)
    case _ => throw new IllegalArgumentException(s"Unsupported evaluator $name!")
  }
}
