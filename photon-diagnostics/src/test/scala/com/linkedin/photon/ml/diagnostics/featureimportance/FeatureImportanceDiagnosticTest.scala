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

import breeze.linalg.DenseVector
import org.testng.annotations._

import com.linkedin.photon.ml.diagnostics.ModelDiagnostic
import com.linkedin.photon.ml.model.Coefficients
import com.linkedin.photon.ml.stat.BasicStatisticalSummary
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import com.linkedin.photon.ml.supervised.regression.LinearRegressionModel

/**
 * Check feature importance diagnostics
 */
class FeatureImportanceDiagnosticTest {

  import org.testng.Assert._

  private def generateModel(size: Int): (Map[String, Int], GeneralizedLinearModel, BasicStatisticalSummary) = {
    val count = 1000L
    val features = DenseVector((1 to size).map(_.toDouble).toArray)
    val coefficients = Coefficients(features, variancesOption = None)
    val summary = new BasicStatisticalSummary(features, features, count, DenseVector.ones[Double](size) * count.toDouble, features, features, features, features, features)
    val model = new LinearRegressionModel(coefficients)
    val featureIdx = (1 to size).map(x => (s"FEATURE_$x", x - 1)).toMap[String, Int]

    println(s"Generated model with target size $size (features: ${features.length}, feature idx mapping ${featureIdx.size})")

    (featureIdx, model, summary)
  }

  /**
   * Generate all the stuff we need to know about a "small" (i.e. fewer features than [[AbstractFeatureImportanceDiagnostic.MAX_RANKED_FEATURES]])
    *
    * @return
   */
  private def generateSmallModel(): (Map[String, Int], GeneralizedLinearModel, BasicStatisticalSummary) = {
    generateModel(AbstractFeatureImportanceDiagnostic.MAX_RANKED_FEATURES / 2)
  }

  /**
   * Generate all the stuff we need to know about a "large" (i.e. several times more features than [[AbstractFeatureImportanceDiagnostic.MAX_RANKED_FEATURES]])
    *
    * @return
   */
  private def generateLargeModel(): (Map[String, Int], GeneralizedLinearModel, BasicStatisticalSummary) = {
    generateModel(2 * AbstractFeatureImportanceDiagnostic.MAX_RANKED_FEATURES)
  }

  @DataProvider
  def generateHappyPathScenarios(): Array[Array[Object]] = {
    val diagnosticFactories = Seq(
      (x: Map[String, Int]) => new VarianceFeatureImportanceDiagnostic(x),
      (x: Map[String, Int]) => new ExpectedMagnitudeFeatureImportanceDiagnostic(x))

    Seq(generateSmallModel(), generateLargeModel()).flatMap(x => {
      val (featureNames, model, summary) = x
      diagnosticFactories.flatMap(factory => {
        Seq(
          Array(factory(featureNames), featureNames, model, None),
          Array(factory(featureNames), featureNames, model, Some(summary)))
      })
    }).toArray
  }

  @Test(dataProvider = "generateHappyPathScenarios")
  def checkHappyPath(diagnostic: ModelDiagnostic[GeneralizedLinearModel, FeatureImportanceReport],
                     featureIndices: Map[String, Int],
                     model: GeneralizedLinearModel,
                     summary: Option[BasicStatisticalSummary]): Unit = {
    val report = diagnostic.diagnose(model, null, summary)
    val expectedSize = math.min(featureIndices.size, AbstractFeatureImportanceDiagnostic.MAX_RANKED_FEATURES)
    val expectedRankImportanceSamples = AbstractFeatureImportanceDiagnostic.NUM_IMPORTANCE_FRACTILES + 1

    assertEquals(report.rankToImportance.size, expectedRankImportanceSamples, "Number of (rank, importance) tuples matches expectations")
    assertEquals(report.featureImportance.size, expectedSize, "Expected number of (feature -> description) tuples")
    // Because of how the test data was constructed, we know what these should be
    val importances = if (summary.isDefined) {
      model.coefficients.means :* model.coefficients.means
    } else {
      model.coefficients.means
    }
    val expectedImportances = Set(importances.toArray.sorted.reverse.take(expectedSize): _*)
    val actualImportances = Set(report.featureImportance.values.map(_._2).toSeq: _*)
    val missingExpected = actualImportances -- expectedImportances
    val extraActual = expectedImportances -- actualImportances
    assertEquals(missingExpected.size, 0, "No expected importances are missing")
    assertEquals(extraActual.size, 0, "No unexpected extra importances")
  }
}
