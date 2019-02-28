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
import com.linkedin.photon.ml.projector.vector.VectorProjector
import com.linkedin.photon.ml.util.VectorUtils

/**
 * Local dataset implementation.
 *
 * @note One design concern is whether to store the local data as a [[Map]] or an [[Array]] (high sort cost, but low
 *       merge cost vs. no sort cost but high merge cost). Currently, we use an [[Array]] since the data is only sorted
 *       once, and used as the base for all other data/score [[Array]]s.
 *
 * @param dataPoints Local data points consists of (globalId, labeledPoint) pairs
 */
protected[ml] case class LocalDataset(dataPoints: Array[(UniqueSampleId, LabeledPoint)]) {

  require(
    dataPoints.length > 0,
    "Cannot create LocalDataset with empty data array")

  val numDataPoints: Int = dataPoints.length
  val numFeatures: Int = dataPoints.head._2.features.length

  /**
   *
   * @return
   */
  def getLabels: Array[(UniqueSampleId, Double)] = dataPoints.map { case (uid, labeledPoint) =>
    (uid, labeledPoint.label)
  }

  /**
   *
   * @return
   */
  def getWeights: Array[(UniqueSampleId, Double)] = dataPoints.map { case (uid, labeledPoint) =>
    (uid, labeledPoint.weight)
  }

  /**
   *
   * @return
   */
  def getOffsets: Array[(UniqueSampleId, Double)] = dataPoints.map { case (uid, labeledPoint) =>
    (uid, labeledPoint.offset)
  }

  /**
   *
   * @return
   */
  def getUniqueIds: Array[UniqueSampleId] = dataPoints.map(_._1)

  /**
   * Add the residual scores to the offsets.
   *
   * @param residualScores The residual scores
   * @return The [[LocalDataset]] with updated offsets
   */
  def addScoresToOffsets(residualScores: Array[(UniqueSampleId, Double)]): LocalDataset = {

    val updatedDataPoints = dataPoints
      .zip(residualScores)
      .map { case ((dataId, LabeledPoint(label, features, offset, weight)), (residualScoreId, residualScoreDatum)) =>

        require(residualScoreId == dataId, s"residual score Id ($residualScoreId) and data Id ($dataId) don't match!")

        (dataId, LabeledPoint(label, features, residualScoreDatum + offset, weight))
      }

    LocalDataset(updatedDataPoints)
  }

  /**
   * Project the features of the underlying [[dataPoints]] from the original space to a projected space.
   *
   * @param projector A [[VectorProjector]] to project the feature data
   * @return A new [[LocalDataset]] with features in the projected space
   */
  def projectForward(projector: VectorProjector): LocalDataset = {

    val projectedDataPoints = dataPoints.map { case (uniqueId, LabeledPoint(label, features, offset, weight)) =>
      (uniqueId, LabeledPoint(label, projector.projectForward(features), offset, weight))
    }

    LocalDataset(projectedDataPoints)
  }

  /**
   * Project the features of the underlying [[dataPoints]] from the projected space to a original space.
   *
   * @param projector A [[VectorProjector]] to project the feature data
   * @return A new [[LocalDataset]] with features in the original space
   */
  def projectBackward(projector: VectorProjector): LocalDataset = {

    val projectedDataPoints = dataPoints.map { case (uniqueId, LabeledPoint(label, features, offset, weight)) =>
      (uniqueId, LabeledPoint(label, projector.projectBackward(features), offset, weight))
    }

    LocalDataset(projectedDataPoints)
  }

  /**
   * Filter features by Pearson correlation score.
   *
   * @param numFeaturesToKeep The number of features to keep
   * @return The filtered dataset
   */
  def filterFeaturesByPearsonCorrelationScore(numFeaturesToKeep: Int): LocalDataset = {

    val numActiveFeatures: Int = dataPoints.flatMap(_._2.features.activeKeysIterator).toSet.size

    if (numFeaturesToKeep < numActiveFeatures) {
      val labelAndFeatures = dataPoints.map { case (_, labeledPoint) => (labeledPoint.label, labeledPoint.features) }
      val pearsonScores = LocalDataset.stableComputePearsonCorrelationScore(labelAndFeatures)

      val filteredFeaturesIndexSet = pearsonScores
        .toArray
        .sortBy { case (_, score) => math.abs(score) }
        .takeRight(numFeaturesToKeep)
        .map(_._1)
        .toSet

      val filteredActivities = dataPoints.map { case (id, LabeledPoint(label, features, offset, weight)) =>

        val filteredFeatures = LocalDataset.filterFeaturesWithFeatureIndexSet(features, filteredFeaturesIndexSet)

        (id, LabeledPoint(label, filteredFeatures, offset, weight))
      }

      LocalDataset(filteredActivities)
    } else {
      this
    }
  }
}

object LocalDataset {
  /**
   * Factory method for LocalDataset.
   *
   * @param dataPoints The array of underlying data
   * @param isSortedByFirstIndex Whether or not to sort the data by global ID
   * @return A new LocalDataset
   */
  protected[ml] def apply(
      dataPoints: Array[(UniqueSampleId, LabeledPoint)],
      isSortedByFirstIndex: Boolean): LocalDataset = {

    if (isSortedByFirstIndex) {
      LocalDataset(dataPoints)
    } else {
      LocalDataset(dataPoints.sortBy(_._1))
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
   * Compute Pearson correlation scores using a numerically stable algorithm.
   *
   * @param labelAndFeatures An array of (label, feature) tuples
   * @return The Pearson correlation scores for each tuple
   */
  protected[ml] def stableComputePearsonCorrelationScore(
    labelAndFeatures: Array[(Double, Vector[Double])]): Map[Int, Double] = {

    val featureMeans = mutable.Map[Int, Double]()
    val featureUnscaledVars = mutable.Map[Int, Double]()
    var labelMean = 0.0
    var labelUnscaledVariance = 0.0
    val unscaledCovariances = mutable.Map[Int, Double]()
    var interceptAdded = false
    var numSamples = 0

    labelAndFeatures.foreach { case (label, features) =>
      numSamples += 1

      val deltaLabel = label - labelMean
      labelMean += deltaLabel / numSamples
      labelUnscaledVariance += deltaLabel * (label - labelMean)

      // Note that, if there is duplicated keys in the feature vector, then the following Pearson correlation scores
      // calculation will screw up
      features.iterator.foreach { case (key, value) =>
        val prevFeatureMean = featureMeans.getOrElse(key, 0.0)
        val deltaFeature = value - prevFeatureMean
        val featureMean = prevFeatureMean + deltaFeature / numSamples

        val prevFeatureUnscaledVar = featureUnscaledVars.getOrElse(key, 0.0)
        val featureUnscaledVar = prevFeatureUnscaledVar + deltaFeature * (value - featureMean)

        val prevCovariance = unscaledCovariances.getOrElse(key, 0.0)
        val unscaledCovariance = prevCovariance + deltaFeature * deltaLabel * (numSamples - 1) / numSamples

        featureMeans.update(key, featureMean)
        featureUnscaledVars.update(key, featureUnscaledVar)
        unscaledCovariances.update(key, unscaledCovariance)
      }
    }

    val labelStd = math.sqrt(labelUnscaledVariance)

    featureMeans
      .iterator
      .map { case (key, featureMean) =>
        val featureStd = math.sqrt(featureUnscaledVars(key))
        val covariance = unscaledCovariances(key)

        // When the standard deviation of the feature is close to 0 we treat it as the intercept term.
        val score = if (featureStd < math.sqrt(numSamples) * MathConst.EPSILON) {
          // Note that if the mean and standard deviation are equal to zero, it either means that the feature is constant
          if (featureMean == 1.0 && !interceptAdded) {
            interceptAdded = true
            1.0
          } else {
            0.0
          }
        } else {
          covariance / (labelStd * featureStd + MathConst.EPSILON)
        }

        require(math.abs(score) <= 1 + MathConst.EPSILON,
          s"Computed pearson correlation score is $score, while the score's magnitude should be less than 1. " +
            s"(Diagnosis:\n" +
            s"featureKey=$key\n" +
            s"featureStd=$featureStd\n" +
            s"labelStd=$labelStd\n" +
            s"covariance=$covariance\n" +
            s"numSamples=$numSamples\n" +
            s"labelAndFeatures used to compute Pearson correlation score:\n${labelAndFeatures.mkString("\n")}})")

        (key, score)
      }
      .toMap
  }

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
