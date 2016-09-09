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

import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.data.GameDatum

/**
 * Interface for evaluation implementations at the [[RDD]] level
 */
protected[ml] trait Evaluator {

  /**
   * The type of the evaluator
   */
  protected val evaluatorType: EvaluatorType

  /**
    * Evaluate the scores of the model
    *
    * @param scores the scores to evaluate
    * @return score metric value
    */
  def evaluate(scores: RDD[(Long, Double)]): Double

  /**
    * Determine the best between two scores returned by the evaluator. In some cases, the better score is higher
    * (e.g. AUC) and in others, the better score is lower (e.g. RMSE).
    *
    * @param score1 the first score to compare
    * @param score2 the second score to compare
    * @return true if the first score is better than the second
    */
  def betterThan(score1: Double, score2: Double): Boolean

  def getEvaluatorName: String = evaluatorType.name
}

object Evaluator {

  def buildEvaluator(evaluatorType: EvaluatorType, gameDataSet: RDD[(Long, GameDatum)]): Evaluator = {
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
      case PrecisionAtK(k, documentIdName) =>
        val documentIds = gameDataSet.mapValues(_.idTypeToValueMap(documentIdName))
        new PrecisionAtKEvaluator(k, labelAndOffsetAndWeights, documentIds, documentIdName)
      case _ => throw new UnsupportedOperationException(s"Unsupported evaluator type: $evaluatorType")
    }
  }
}
