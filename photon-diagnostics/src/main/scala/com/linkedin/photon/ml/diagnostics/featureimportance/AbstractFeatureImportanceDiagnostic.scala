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
package com.linkedin.photon.ml.diagnostics.featureimportance

import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.diagnostics.ModelDiagnostic
import com.linkedin.photon.ml.stat.BasicStatisticalSummary
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import org.apache.spark.rdd.RDD

/**
  * Common base class for various notions of feature importance in GLM
  *
  * @param modelNameToIndex Map of (encoded name/term &rarr; feature index)
  */
abstract class AbstractFeatureImportanceDiagnostic(protected val modelNameToIndex: Map[String, Int])
  extends ModelDiagnostic[GeneralizedLinearModel, FeatureImportanceReport] {

  import AbstractFeatureImportanceDiagnostic._

  def diagnose(
      model: GeneralizedLinearModel,
      data: RDD[LabeledPoint],
      summary: Option[BasicStatisticalSummary]): FeatureImportanceReport = {

    val importanceMeasure = getImportanceDescription(summary)

    val importances = getImportances(model, summary).toList.sortBy(_._3).reverse

    val rankToImportance = getRankToImportance(importances)
    val chosenFeatures = importances.take(MAX_RANKED_FEATURES)
    val featureToDescription = chosenFeatures.map(x => {
      val (id, idx, imp) = x
      (id, (idx, imp, describeFeature(id, idx, imp, model, summary)))
    }).toMap

    new FeatureImportanceReport(
      importanceType = getImportanceType,
      importanceDescription = importanceMeasure,
      featureImportance = featureToDescription,
      rankToImportance = rankToImportance)
  }

  // TODO: Abstract methods need additional documentation

  /**
    * @param summary Summary of the data (if available)
    * @return Description of how the importance scores are calculated
    */
  protected def getImportanceDescription(summary: Option[BasicStatisticalSummary]): String

  /**
    *
    * @param model
    * @param summary
    * @return Iterable of ((name, term), index, importance) tuples. Importance should be computed in such a way that
    *         bigger is better (i.e. sorting in decreasing order implies decreasing importance)
    */
  protected def getImportances(
    model: GeneralizedLinearModel,
    summary: Option[BasicStatisticalSummary]): Iterable[((String, String), Int, Double)]

  /**
    * @return The measure of importance is being used. This is in contrast with the importance description. The intent
    *         is that while many notions of feature importance may result in the same <em>description</em> of feature
    *         importance (e.g. they fall back to coefficient magnitude when no summary is available), the <em>type</em>
    *         should be unique on a per-child-class basis.
    */
  protected def getImportanceType: String

  private def getRankToImportance(sortedByImportance: Seq[((String, String), Int, Double)]): Map[Double, Double] = {
    val indices = (0 to NUM_IMPORTANCE_FRACTILES)
      .map(x => math.min(sortedByImportance.size - 1, x * sortedByImportance.size / MAX_RANKED_FEATURES))

    val fractiles = (0 to NUM_IMPORTANCE_FRACTILES)
      .map(100.0 * _ / NUM_IMPORTANCE_FRACTILES)

    val imp = indices.map(x => sortedByImportance(x)._3)
    fractiles.zip(imp).toMap
  }

  private def describeFeature(
      id: (String, String),
      idx: Int,
      importance: Double,
      model: GeneralizedLinearModel,
      summary: Option[BasicStatisticalSummary]): String = {

    val basic = f"Feature (name=[${id._1}], term=[${id._2}]) importance = [$importance%.03f], " +
      f"coefficient = [${model.coefficients.means(idx)}%.06g]"

    val extended = summary match {
      case Some(sum) =>
        f" min=[${sum.min(idx)}], mean=[${sum.mean(idx)}], max=[${sum.max(idx)}], variance=[${sum.variance(idx)}]"
      case None => ""
    }

    basic + extended
  }
}

object AbstractFeatureImportanceDiagnostic {
  val MAX_RANKED_FEATURES = 50
  val NUM_IMPORTANCE_FRACTILES = 100
}
