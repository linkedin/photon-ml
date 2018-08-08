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
package com.linkedin.photon.ml.data

import scala.collection.mutable

import breeze.linalg.Vector

import com.linkedin.photon.ml.Types.UniqueSampleId
import com.linkedin.photon.ml.constants.MathConst
import com.linkedin.photon.ml.projector.Projector
import com.linkedin.photon.ml.util.VectorUtils

/**
 * Local data set implementation.
 *
 * @note One design concern is whether to store the local data as a [[Map]] or an [[Array]] (high sort cost, but low
 *       merge cost vs. no sort cost but high merge cost). Currently, we use an [[Array]] since the data is only sorted
 *       once, and used as the base for all other data/score [[Array]]s.
 *
 * @param dataPoints Local data points consists of (globalId, labeledPoint) pairs
 */
protected[ml] case class LocalDataSet(dataPoints: Array[(UniqueSampleId, LabeledPoint)]) {

  require(
    dataPoints.length > 0,
    "Cannot create LocalDataSet with empty data array")

  val numDataPoints: Int = dataPoints.length
  val numFeatures: Int = dataPoints.head._2.features.length
  val numActiveFeatures: Int = dataPoints.flatMap(_._2.features.activeKeysIterator).toSet.size

  /**
   *
   * @return
   */
  def getLabels: Array[(UniqueSampleId, Double)] = dataPoints.map { case (uid, labeledPoint) => (uid, labeledPoint.label) }

  /**
   *
   * @return
   */
  def getWeights: Array[(UniqueSampleId, Double)] = dataPoints.map { case (uid, labeledPoint) => (uid, labeledPoint.weight) }

  /**
   *
   * @return
   */
  def getOffsets: Array[(UniqueSampleId, Double)] = dataPoints.map { case (uid, labeledPoint) => (uid, labeledPoint.offset) }

  /**
   *
   * @return
   */
  def getUniqueIds: Array[UniqueSampleId] = dataPoints.map(_._1)

  /**
   * Add the residual scores to the offsets.
   *
   * @param residualScores The residual scores
   * @return The [[LocalDataSet]] with updated offsets
   */
  def addScoresToOffsets(residualScores: Array[(UniqueSampleId, Double)]): LocalDataSet = {

    val updatedDataPoints = dataPoints
      .zip(residualScores)
      .map { case ((dataId, LabeledPoint(label, features, offset, weight)), (residualScoreId, residualScoreDatum)) =>

        require(residualScoreId == dataId, s"residual score Id ($residualScoreId) and data Id ($dataId) don't match!")

        (dataId, LabeledPoint(label, features, residualScoreDatum + offset, weight))
      }

    LocalDataSet(updatedDataPoints)
  }

  /**
   * Project the features of the underlying [[dataPoints]] from the original space to the projected
   * (usually with lower dimension) space.
   *
   * @param projector The projector
   * @return The [[LocalDataSet]] with projected features
   */
  def projectFeatures(projector: Projector): LocalDataSet = {

    val projectedDataPoints = dataPoints.map { case (uniqueId, LabeledPoint(label, features, offset, weight)) =>
      (uniqueId, LabeledPoint(label, projector.projectFeatures(features), offset, weight))
    }

    LocalDataSet(projectedDataPoints)
  }

  /**
   * Filter features by binomial ratio confidence intervals.
   *
   * @param globalFeatureInstances The global instances with the features present
   * @param globalPositiveInstances The global positive instances with the features present
   * @param binaryIndices The binary feature columns indices
   * @param nonBinaryIndices The binary feature columns indices
   * @param intervalBound The lower bound threshold of the confidence interval used to filter features
   * @param zScore The Z-score for the chosen two-tailed confidence level
   * @return The filtered dataset
   */
  def filterFeaturesByRatioCIBound(
      globalFeatureInstances: Array[Double],
      globalPositiveInstances: Array[Double],
      binaryIndices: Set[Int],
      nonBinaryIndices: Set[Int],
      intervalBound: Double = 1.0,
      zScore: Double = 2.575): LocalDataSet = {

    val labelAndFeatures = dataPoints.map { case (_, labeledPoint) => (labeledPoint.label, labeledPoint.features) }
    val lowerBounds = LocalDataSet.computeRatioCILowerBound(
      labelAndFeatures,
      globalFeatureInstances,
      globalPositiveInstances,
      binaryIndices,
      numFeatures,
      zScore)
    val filteredBinaryFeaturesIndexSet = lowerBounds.filter(_._2 > intervalBound).keySet
    val filteredFeaturesIndexSet = filteredBinaryFeaturesIndexSet ++ nonBinaryIndices

    val filteredActivities = dataPoints.map { case (id, LabeledPoint(label, features, offset, weight)) =>

      val filteredFeatures = LocalDataSet.filterFeaturesWithFeatureIndexSet(features, filteredFeaturesIndexSet)

      (id, LabeledPoint(label, filteredFeatures, offset, weight))
    }

    LocalDataSet(filteredActivities)
  }

  /**
   * Filter features by Pearson correlation score.
   *
   * @param numFeaturesToKeep The number of features to keep
   * @return The filtered dataset
   */
  def filterFeaturesByPearsonCorrelationScore(numFeaturesToKeep: Int): LocalDataSet =
    if (numFeaturesToKeep < numActiveFeatures) {

      val labelAndFeatures = dataPoints.map { case (_, labeledPoint) => (labeledPoint.label, labeledPoint.features) }
      val pearsonScores = LocalDataSet.computePearsonCorrelationScore(labelAndFeatures)

      val filteredFeaturesIndexSet = pearsonScores
        .toArray
        .sortBy { case (_, score) => math.abs(score) }
        .takeRight(numFeaturesToKeep)
        .map(_._1)
        .toSet

      val filteredActivities = dataPoints.map { case (id, LabeledPoint(label, features, offset, weight)) =>

        val filteredFeatures = LocalDataSet.filterFeaturesWithFeatureIndexSet(features, filteredFeaturesIndexSet)

        (id, LabeledPoint(label, filteredFeatures, offset, weight))
      }

      LocalDataSet(filteredActivities)

    } else {
      this
    }
}

object LocalDataSet {
  /**
   * Factory method for LocalDataSet.
   *
   * @param dataPoints The array of underlying data
   * @param isSortedByFirstIndex Whether or not to sort the data by global ID
   * @return A new LocalDataSet
   */
  val EPSILON = 0.5

  protected[ml] def apply(
      dataPoints: Array[(UniqueSampleId, LabeledPoint)],
      isSortedByFirstIndex: Boolean): LocalDataSet = {

    if (isSortedByFirstIndex) {
      LocalDataSet(dataPoints)
    } else {
      LocalDataSet(dataPoints.sortBy(_._1))
    }
  }

  /**
   * Filter features by feature index.
   *
   * @param features The original feature set
   * @param featureIndexSet The feature index set
   * @return The filtered feature vector
   */
  private def filterFeaturesWithFeatureIndexSet(
      features: Vector[Double],
      featureIndexSet: Set[Int]): Vector[Double] = {

    val result = VectorUtils.zeroOfSameType(features)

    features.activeIterator.foreach { case (key, value) =>
      if (featureIndexSet.contains(key)) {
        result(key) = value
      }
    }

    result
  }

  /**
   * Compute feature ratio confidence interval lower bounds.
   *
   * @param labelAndFeatures An [[Array]] of (label, feature vector) tuples
   * @param zScore The Z-score for the chosen two-tailed confidence level
   * @param globalFeatureInstances The global instances with the features present
   * @param globalPositiveInstances The global positive instances with the features present
   * @param binaryIndices The binary feature columns indices
   * @param epsilon The constant used to compute for extreme case of ratio modeling
   * @return the lowerBounds for feature columns
   */
  protected[ml] def computeRatioCILowerBound(
      labelAndFeatures: Array[(Double, Vector[Double])],
      globalFeatureInstances: Array[Double],
      globalPositiveInstances: Array[Double],
      binaryIndices: Set[Int],
      numFeatures: Int,
      zScore: Double): Map[Int, Double] = {

    val n = globalFeatureInstances
    val y = globalPositiveInstances

    val m = labelAndFeatures.map(_._2).reduce(_ + _)
    val x = labelAndFeatures
      .filter(_._1 > MathConst.POSITIVE_RESPONSE_THRESHOLD)
      .map(_._2)
      .foldLeft(Vector.zeros[Double](numFeatures))(_ + _)

    binaryIndices
      .map { key =>
        val x_col = x(key)
        val m_col = m(key)
        val y_col = y(key)
        val n_col = n(key)

        val lowerBound = if (y_col == 0.0 || y_col == n_col) {
          0D
        } else {
          val (t, variance) = computeTAndVariance(math.max(x_col, EPSILON), m_col, y_col, n_col)

          if (t < 1D) {
            1D / computeUpperBound(t, variance, zScore)
          } else {
            computeLowerBound(t, variance, zScore)
          }
        }

        (key, lowerBound)
      }
      .toMap
  }

  /**
   * Compute t value and variance for ratio modelling.
   *
   * @param x The count for f_i == 1 and label == 1 in the local population
   * @param m The count for f_i == 1 in the local population
   * @param y The count for f_i == 1 and label == 1 in the global population
   * @param n The count for f_i == 1 in the global population
   * @return The mean and variance for ratio t
   */
  protected[ml] def computeTAndVariance(x: Double, m: Double, y: Double, n: Double): (Double, Double) =
    if (m == 0.0 || n == 0.0 || y == 0.0 || x == 0.0) {
      (0.0, 0.0)
    } else {
      val t = (x / m) / (y / n)
      val variance = 1.0 / x - 1.0 / m + 1.0 / y - 1.0 / n

      (t, variance)
    }

  /**
   * Compute the confidence interval lowerbound for ratio modelling.
   *
   * @param t The value of ratio
   * @param variance The variance of the ratio
   * @param zScore The Z-score for the chosen two-tailed confidence level
   * @return The lowerbound for ratio t
   */
  protected[ml] def computeLowerBound(
      t: Double,
      variance: Double,
      zScore: Double): Double =
    t * math.exp(-math.sqrt(variance) * zScore)

  /**
   * Compute the confidence interval upperbound for ratio modelling.
   *
   * @param t The value of ratio
   * @param variance The variance of the ratio
   * @param zScore The Z-score for the chosen two-tailed confidence level
   * @return The upperbound for ratio t
   */
  protected[ml] def computeUpperBound(
      t: Double,
      variance: Double,
      zScore: Double): Double =
    t * math.exp(math.sqrt(variance) * zScore)

  /**
   * Compute Pearson correlation scores.
   *
   * @param labelAndFeatures An array of (label, feature) tuples
   * @return The Pearson correlation scores for each tuple
   */
  protected[ml] def computePearsonCorrelationScore(
      labelAndFeatures: Array[(Double, Vector[Double])]): Map[Int, Double] = {

    val featureLabelProductSums = mutable.Map[Int, Double]()
    val featureFirstOrderSums = mutable.Map[Int, Double]()
    val featureSecondOrderSums = mutable.Map[Int, Double]()
    var labelFirstOrderSum = 0.0
    var labelSecondOrderSum = 0.0
    var numSamples = 0
    var interceptAdded = false

    labelAndFeatures.foreach { case (label, features) =>
      numSamples += 1
      labelFirstOrderSum += label
      labelSecondOrderSum += label * label
      // Note that, if there is duplicated keys in the feature vector, then the following Pearson correlation scores
      // calculation will screw up
      features.activeIterator.foreach { case (key, value) =>
        featureFirstOrderSums.update(key, featureFirstOrderSums.getOrElse(key, 0.0) + value)
        featureSecondOrderSums.update(key, featureSecondOrderSums.getOrElse(key, 0.0) + value * value)
        featureLabelProductSums.update(key, featureLabelProductSums.getOrElse(key, 0.0) + value * label)
      }
    }

    featureFirstOrderSums
      .keySet
      .map { key =>
        val featureFirstOrderSum = featureFirstOrderSums(key)
        val featureSecondOrderSum = featureSecondOrderSums(key)
        val featureLabelProductSum = featureLabelProductSums(key)
        val numerator = numSamples * featureLabelProductSum - featureFirstOrderSum * labelFirstOrderSum
        val std = math.sqrt(math.abs(numSamples * featureSecondOrderSum - featureFirstOrderSum * featureFirstOrderSum))
        val denominator = std * math.sqrt(numSamples * labelSecondOrderSum - labelFirstOrderSum * labelFirstOrderSum)

        // When the standard deviation of the feature is close to 0, we treat it as the intercept term
        val score = if (std < MathConst.EPSILON) {
          if (interceptAdded) {
            0.0
          } else {
            interceptAdded = true
            1.0
          }
        } else {
          numerator / (denominator + MathConst.EPSILON)
        }

        require(math.abs(score) <= 1 + MathConst.EPSILON,
          s"Computed pearson correlation score is $score, while the score's magnitude should be less than 1. " +
          s"(Diagnosis:\n" +
          s"numerator=$numerator\n" +
          s"denominator=$denominator\n" +
          s"numSamples=$numSamples\n" +
          s"featureFirstOrderSum=$featureFirstOrderSum\n" +
          s"featureSecondOrderSum=$featureSecondOrderSum\n" +
          s"featureLabelProductSum=$featureLabelProductSum\n" +
          s"labelFirstOrderSum=$labelFirstOrderSum\n" +
          s"labelSecondOrderSum=$labelSecondOrderSum\n" +
          s"labelAndFeatures used to compute Pearson correlation score:\n${labelAndFeatures.mkString("\n")}})")

        (key, score)
      }
      .toMap
  }
}
