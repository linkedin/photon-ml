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

import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.data.GameDatum
import com.linkedin.photon.ml.evaluation.EvaluatorType._

object EvaluatorFactory {
  /**
   * Factory for different types of [[Evaluator]].
   *
   * @param evaluatorType The type of the evaluator
   * @param gameDataSet A [[RDD]] of (uniqueId: [[Long]], GAMEDatum: [[GAMEDatum]]), which are usually the
   *                    validation/test data, used to construct the evaluator
   * @return The evaluator
   */
  protected[ml] def buildEvaluator(evaluatorType: EvaluatorType, gameDataSet: RDD[(Long, GameDatum)]): Evaluator = {
    val labelAndOffsetAndWeights = gameDataSet.mapValues(gameData =>
      (gameData.response, gameData.offset, gameData.weight)
    )

    evaluatorType match {
      case AUC => new AreaUnderROCCurveEvaluator(labelAndOffsetAndWeights)
      case RMSE => new RMSEEvaluator(labelAndOffsetAndWeights)
      case PoissonLoss => new PoissonLossEvaluator(labelAndOffsetAndWeights)
      case LogisticLoss => new LogisticLossEvaluator(labelAndOffsetAndWeights)
      case SmoothedHingeLoss => new SmoothedHingeLossEvaluator(labelAndOffsetAndWeights)
      case SquaredLoss => new SquaredLossEvaluator(labelAndOffsetAndWeights)
      case ShardedPrecisionAtK(k, idType) =>
        val ids = gameDataSet.mapValues(_.idTypeToValueMap(idType))
        new ShardedPrecisionAtKEvaluator(k, idType, ids, labelAndOffsetAndWeights)
      case ShardedAUC(idType) =>
        val ids = gameDataSet.mapValues(_.idTypeToValueMap(idType))
        new ShardedAreaUnderROCCurveEvaluator(idType, ids, labelAndOffsetAndWeights)
      case _ => throw new UnsupportedOperationException(s"Unsupported evaluator type: $evaluatorType")
    }
  }
}
