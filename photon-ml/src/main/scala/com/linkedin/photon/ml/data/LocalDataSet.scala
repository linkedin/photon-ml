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
package com.linkedin.photon.ml.data

import java.util.Random

import scala.collection.{Map, Set, mutable}
import scala.reflect.ClassTag

import breeze.linalg.{SparseVector, Vector}

import com.linkedin.photon.ml.constants.MathConst
import com.linkedin.photon.ml.projector.Projector

/**
 * Local dataset implementation
 *
 * @param dataPoints Local data points consists of (uniqueId, labeledPoint) pairs
 * @author xazhang
 * @todo Use Array or Map to represent the local data structure?
 *       Array: overhead in sorting the entries by key
 *       Map: overhead in accessing value by key
 */
case class LocalDataSet(dataPoints: Array[(Long, LabeledPoint)]) {

  val numDataPoints = dataPoints.length
  val numFeatures = if (numDataPoints > 0) dataPoints.head._2.features.length else 0
  val numActiveFeatures = if (numDataPoints > 0) dataPoints.flatMap(_._2.features.activeKeysIterator).toSet.size else 0

  def getLabels: Array[(Long, Double)] = dataPoints.map { case (uid, labeledPoint) => (uid, labeledPoint.label) }

  def getWeights: Array[(Long, Double)] = dataPoints.map { case (uid, labeledPoint) => (uid, labeledPoint.weight) }

  def getOffsets: Array[(Long, Double)] = dataPoints.map { case (uid, labeledPoint) => (uid, labeledPoint.offset) }

  def getGlobalIds: Array[Long] = dataPoints.map(_._1)

  /**
   * Add the residual scores to the offsets
   *
   * @param residualScores The residual scores
   * @return The [[LocalDataSet]] with updated offsets
   */
  def addScoresToOffsets(residualScores: Array[(Long, Double)]): LocalDataSet = {
    val updatedDataPoints = dataPoints.zip(residualScores).map {
      case ((dataId, LabeledPoint(label, features, offset, weight)), (residualScoreId, residualScore)) =>
        assert(residualScoreId == dataId, s"residual score Id ($residualScoreId) and data Id ($dataId) don't match!")
        (dataId, LabeledPoint(label, features, residualScore + offset, weight))
    }
    LocalDataSet(updatedDataPoints)
  }

  /**
   * Project the features of the underlying [[dataPoints]] from the original space to the projected
   * (usually with lower dimension) space
   *
   * @param projector The projector
   * @return The [[LocalDataSet]] with projected features
   */
  def projectFeatures(projector: Projector): LocalDataSet = {
    val projectedDataPoints = dataPoints.map { case (uniqueId, LabeledPoint(label, features, offset, weight)) =>
      val projectedFeatures = projector.projectFeatures(features)
      (uniqueId, LabeledPoint(label, projectedFeatures, offset, weight))
    }
    LocalDataSet(projectedDataPoints)
  }

  private lazy val featureIndexCountMap: Map[Int, Int] = {
    dataPoints
      .map(_._2.features)
      .flatMap(_.activeKeysIterator)
      .map((_, 1))
      .groupBy(_._1)
      .map { case (index, counts) => (index, counts.map(_._2).sum) }
  }

  /**
   * Filter features by support
   *
   * @param minNumSupportThreshold minimum support threshold
   * @return filtered dataset
   */
  def filterFeaturesBySupport(minNumSupportThreshold: Int): LocalDataSet = {
    if (minNumSupportThreshold > 0) {
      val filteredFeaturesIndexSet = featureIndexCountMap
          .filter { case (_, count) => count >= minNumSupportThreshold }
          .keySet

      val filteredActivities = dataPoints.map { case (id, LabeledPoint(label, features, offset, weight)) =>
        val filteredFeatures = LocalDataSet.filterFeaturesWithFeatureIndexSet(features, filteredFeaturesIndexSet)
        (id, LabeledPoint(label, filteredFeatures, offset, weight))
      }

      LocalDataSet(filteredActivities)
    } else {
      this
    }
  }

  /**
   * Filter features by Pearson correlation score
   *
   * @param numFeaturesToKeep number of features to keep
   * @return filtered dataset
   */
  def filterFeaturesByPearsonCorrelationScore(numFeaturesToKeep: Int): LocalDataSet = {
    if (numFeaturesToKeep < numActiveFeatures) {
      val labelAndFeatures = dataPoints.map { case (_, labeledPoint) => (labeledPoint.label, labeledPoint.features) }
      val pearsonScores = LocalDataSet.computePearsonCorrelationScore(labelAndFeatures.iterator)

      val filteredFeaturesIndexSet = pearsonScores.toArray
        .sortBy { case (key, score) => math.abs(score) }
        .takeRight(numFeaturesToKeep).map(_._1).toSet

      val filteredActivities = dataPoints.map { case (id, LabeledPoint(label, features, offset, weight)) =>
        val filteredFeatures = LocalDataSet.filterFeaturesWithFeatureIndexSet(features, filteredFeaturesIndexSet)
        (id, LabeledPoint(label, filteredFeatures, offset, weight))
      }

      LocalDataSet(filteredActivities)
    } else {
      this
    }
  }

  /**
   * Reservoir sampling on all samples
   *
   * @param numSamplesToKeep number of samples to keep
   * @return sampled dataset
   */
  protected[ml] def reservoirSamplingOnAllSamples(numSamplesToKeep: Int): LocalDataSet = {
    if (numSamplesToKeep == 0) {
      LocalDataSet(Array())
    } else if (numSamplesToKeep < numDataPoints) {
      val weightRatio = 1.0 * numDataPoints / numSamplesToKeep
      val sampledData = LocalDataSet.reservoirSampling(dataPoints.iterator, numSamplesToKeep)
      LocalDataSet(sampledData.map { case (id, LabeledPoint(label, feature, offset, weight)) =>
        (id, LabeledPoint(label, feature, offset, weight * weightRatio))
      })
    } else {
      this
    }
  }
}

object LocalDataSet {

  def apply(dataPoints: Array[(Long, LabeledPoint)], isSortedByFirstIndex: Boolean): LocalDataSet = {
    if (isSortedByFirstIndex) LocalDataSet(dataPoints)
    else LocalDataSet(dataPoints.sortBy(_._1))
  }

  /**
   * Filter features by feature index
   *
   * @param features the original feature set
   * @param featureIndexSet feature index set
   * @return filtered feature vector
   */
  private def filterFeaturesWithFeatureIndexSet(
      features: Vector[Double],
      featureIndexSet: Set[Int]): SparseVector[Double] = {

    val filteredIndexBuilder = new mutable.ArrayBuilder.ofInt
    val filteredDataBuilder = new mutable.ArrayBuilder.ofDouble
    features.activeIterator.foreach { case (key, value) =>
      if (featureIndexSet.contains(key)) {
        filteredIndexBuilder += key
        filteredDataBuilder += value
      }
    }

    new SparseVector(filteredIndexBuilder.result(), filteredDataBuilder.result(), features.length)
  }

  /**
   * Compute Pearson correlation scores
   *
   * @param labelAndFeatures iterator over label, feature tuples
   * @return correlation scores
   */
  protected[ml] def computePearsonCorrelationScore(
      labelAndFeatures: Iterator[(Double, Vector[Double])]): Map[Int, Double] = {

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

    featureFirstOrderSums.keySet.map { key =>
      val featureFirstOrderSum = featureFirstOrderSums(key)
      val featureSecondOrderSum = featureSecondOrderSums(key)
      val featureLabelProductSum = featureLabelProductSums(key)
      val numerator = numSamples * featureLabelProductSum - featureFirstOrderSum * labelFirstOrderSum
      val denominator = math.sqrt(math.abs(numSamples * featureSecondOrderSum -
          featureFirstOrderSum * featureFirstOrderSum)) *
          math.sqrt(numSamples * labelSecondOrderSum - labelFirstOrderSum * labelFirstOrderSum)

      val score =
      // When the variance of the feature is 0, we treat it as the intercept term
        if (math.abs(numSamples * featureSecondOrderSum - featureFirstOrderSum * featureFirstOrderSum) <
            MathConst.LOW_PRECISION_TOLERANCE_THRESHOLD) {
          if (interceptAdded) 0.0
          else {
            interceptAdded = true
            1.0
          }
        } else {
          numerator / (denominator + MathConst.HIGH_PRECISION_TOLERANCE_THRESHOLD)
        }

      assert(math.abs(score) <= 1 + MathConst.HIGH_PRECISION_TOLERANCE_THRESHOLD,
        s"Computed pearson correlation score is $score, " +
        s"while the score's magnitude should be less than 1. " +
        s"(Diagnosis:\n" +
        s"numerator=$numerator\n" +
        s"denominator=$denominator\n" +
        s"numSamples=$numSamples\n" +
        s"featureFirstOrderSum=$featureFirstOrderSum\n" +
        s"featureSecondOrderSum=$featureSecondOrderSum\n" +
        s"featureLabelProductSum=$featureLabelProductSum\n" +
        s"labelFirstOrderSum=$labelFirstOrderSum\n" +
        s"labelSecondOrderSum=$labelSecondOrderSum)")

      (key, score)
    }.toMap
  }

  /**
   * Reservoir sampling
   *
   * @param dataPoints input data points
   * @param numDataPointsToKeep number of data points to keep
   * @return filtered data points
   */
  protected[ml] def reservoirSampling[T: ClassTag](dataPoints: Iterator[T], numDataPointsToKeep: Int): Array[T] = {
    val reservoir = new Array[T](numDataPointsToKeep)

    // Put the first k elements in the reservoir.
    var i = 0
    while (i < numDataPointsToKeep && dataPoints.hasNext) {
      reservoir(i) = dataPoints.next()
      i += 1
    }

    // If we have consumed all the elements, return them. Otherwise do the replacement.
    if (i < numDataPointsToKeep) {
      // If input size < k, trim the array to return only an array of input size.
      val trimReservoir = new Array[T](i)
      System.arraycopy(reservoir, 0, trimReservoir, 0, i)
      trimReservoir

    } else {
      // If input size > k, continue the sampling process.
      val rand = new Random(MathConst.RANDOM_SEED)
      while (dataPoints.hasNext) {
        val item = dataPoints.next()
        val replacementIndex = rand.nextInt(i)
        if (replacementIndex < numDataPointsToKeep) {
          reservoir(replacementIndex) = item
        }
        i += 1
      }

      reservoir
    }
  }
}
