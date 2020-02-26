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

import org.apache.spark.sql.DataFrame

import com.linkedin.photon.ml.Types.UniqueSampleId
import com.linkedin.photon.ml.constants.DataConst
import com.linkedin.photon.ml.evaluation.EvaluatorType._

/**
 * Factory for [[Evaluator]] object construction.
 */
object EvaluatorFactory {

  /**
   * Construct [[Evaluator]] objects.
   *
   * @param evaluatorType The [[EvaluatorType]]
   * @param gameDataset A [[DataFrame]] of (unique ID, GAME data point, scores) which may be necessary to construct [[MultiEvaluator]]
   *                    objects
   * @return A new [[Evaluator]]
   */
  protected[ml] def buildEvaluator(
      evaluatorType: EvaluatorType,
      gameDataset: DataFrame): Evaluator =
    evaluatorType match {
      case AUC => AreaUnderROCCurveEvaluator

      case AUPR => AreaUnderPRCurveEvaluator

      case RMSE => RMSEEvaluator

      case PoissonLoss => PoissonLossEvaluator

      case LogisticLoss => LogisticLossEvaluator

      case SmoothedHingeLoss => SmoothedHingeLossEvaluator

      case SquaredLoss => SquaredLossEvaluator

      case MultiPrecisionAtK(k, idTag) =>
        val idsRDD = gameDataset.select(DataConst.ID, idTag)
          .rdd.map(row => (row.getAs[UniqueSampleId](0), row.getString(1)))
        new PrecisionAtKMultiEvaluator(k, idTag, idsRDD)

      case MultiAUC(idTag) =>
        val idsRDD = gameDataset.select(DataConst.ID, idTag)
          .rdd.map(row => (row.getAs[UniqueSampleId](0), row.getString(1)))
        new AreaUnderROCCurveMultiEvaluator(idTag, idsRDD)

      case _ =>
        throw new UnsupportedOperationException(s"Unsupported evaluator type: $evaluatorType")
    }
}
