package com.linkedin.photon.ml.diagnostics.featureimportance

import com.linkedin.photon.ml.diagnostics.reporting.LogicalReport

import scala.collection.SortedMap

/**
 * @param importanceType
 * The type of feature importance that is being represented
 * @param importanceDescription
 * Description of how importance is computed
 * @param featureImportance
 * Map of (name, term) &rarr; (feature index, feature importance, feature description)
 * @param rankToImportance
 * Map of importance percentile / rank &rarr; importance @ that rank
 */
case class FeatureImportanceReport(importanceType: String,
                                   importanceDescription: String,
                                   featureImportance: Map[(String, String), (Int, Double, String)],
                                   rankToImportance: Map[Double, Double]) extends LogicalReport