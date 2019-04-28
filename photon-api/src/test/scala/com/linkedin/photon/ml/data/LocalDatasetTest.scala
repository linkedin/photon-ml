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

import java.util.Random

import breeze.linalg.{SparseVector, Vector}
import org.testng.Assert._
import org.testng.annotations.Test

import com.linkedin.photon.ml.constants.MathConst
import com.linkedin.photon.ml.test.CommonTestUtils

/**
 * Unit tests for [[LocalDataset]].
 */
class LocalDatasetTest {

  /**
   * Test the stable Pearson correlation score computation.
   */
  @Test(groups = Array[String]("testPearsonCorrelationScore", "testCore"))
  def testPearsonCorrelationScore(): Unit = {

    // Test input data
    val labels = Array(1.0, 4.0, 6.0, 9.0)
    val features = Array(
      Vector(0.0, 0.0, 2.0), Vector(5.0, 0.0, -3.0), Vector(7.0, 0.0, -8.0), Vector(0.0, 0.0, -1.0))
    val expected = Map(0 -> 0.05564149, 1 -> 0.0, 2 -> -0.40047142)
    val labelAndFeatures = labels.zip(features)
    val computed = LocalDataset.stableComputePearsonCorrelationScore(labelAndFeatures)

    computed.foreach { case (key, value) =>
      assertEquals(
        expected(key),
        value,
        CommonTestUtils.LOW_PRECISION_TOLERANCE,
        s"Computed Pearson correlation score is $value, while the expected value is ${expected(key)}.")
    }
  }

  /**
   * Test the stable Pearson correlation score computation on sparse feature vectors.
   */
  @Test(groups = Array[String]("testPearsonCorrelationScore", "testCore"))
  def testStablePearsonCorrelationScoreOnSparseVector(): Unit = {

    // Test input data
    val labels = Array(1.0, 4.0, 6.0, 9.0)
    val numFeatures = 3

    val features = Array[Vector[Double]](
      new SparseVector[Double](Array(2), Array(2.0), numFeatures),
      new SparseVector[Double](Array(0, 2), Array(5.0, -3.0), numFeatures),
      new SparseVector[Double](Array(0, 2), Array(7.0, -8.0), numFeatures),
      new SparseVector[Double](Array(2), Array(-1.0), numFeatures)
    )

    val expected = Map(0 -> 0.05564149, 1 -> 0.0, 2 -> -0.40047142)

    val labelAndFeatures = labels.zip(features)

    val computed = LocalDataset.stableComputePearsonCorrelationScore(labelAndFeatures)

    computed.foreach { case (key, value) =>
      assertEquals(
        expected(key),
        value,
        CommonTestUtils.LOW_PRECISION_TOLERANCE,
        s"Computed Pearson correlation score is $value, while the expected value is ${expected(key)}.")
    }
  }

  /**
   * Test that the stable Pearson correlation score computation properly recognizes an intercept column.
   */
  @Test(groups = Array[String]("testPearsonCorrelationScore", "testCore"))
  def testPearsonCorrelationScoreForIntercept(): Unit = {

    // Test input data
    val labels = Array(1.0, 4.0, 6.0, 9.0)
    val features = Array(
      Vector(0.0, 0.0, 1.0, 2.0),
      Vector(5.0, 0.0, 1.0, -3.0),
      Vector(7.0, 0.0, 1.0, -8.0),
      Vector(0.0, 0.0, 1.0, -1.0)
    )
    val expected = Map(0 -> 0.05564149, 1 -> 0.0, 2 -> 1.0, 3 -> -0.40047142)
    val labelAndFeatures = labels.zip(features)
    val computed = LocalDataset.stableComputePearsonCorrelationScore(labelAndFeatures)

    computed.foreach { case (key, value) =>
      assertEquals(
        expected(key),
        value,
        CommonTestUtils.LOW_PRECISION_TOLERANCE,
        s"Computed Pearson correlation score is $value, while the expected value is ${expected(key)}.")
    }
  }

  /**
   * Test the stable Pearson correlation score numerical stability.
   */
  @Test(groups = Array[String]("testPearsonCorrelationScore", "testCore"))
  def testStablePearsonCorrelationScoreStability(): Unit = {

    // Test input data: this is a pathological example in which a naive algorithm would fail due to numerical
    // unstability.
    val labels = Array(10000000.0, 10000000.1, 10000000.2)
    val features = Array(Vector(0.0), Vector(0.1), Vector(0.2))

    val expected = Map(0 -> 1.0)

    val labelAndFeatures = labels.zip(features)
    val computed = LocalDataset.stableComputePearsonCorrelationScore(labelAndFeatures)

    computed.foreach { case (key, value) =>
      assertEquals(
        expected(key),
        value,
        CommonTestUtils.LOW_PRECISION_TOLERANCE,
        s"Computed Pearson correlation score is $value, while the expected value is ${expected(key)}.")
    }
  }

  /**
   * Test feature filtering using the stable Pearson correlation score.
   */
  @Test(dependsOnGroups = Array[String]("testPearsonCorrelationScore", "testCore"))
  def testFilterFeaturesByPearsonCorrelationScore(): Unit = {

    val numSamples = 10
    val random = new Random(MathConst.RANDOM_SEED)
    val labels = Array.fill(numSamples)(if (random.nextDouble() > 0.5) 1.0 else -1.0)
    val numFeatures = 10
    // Each data point has 10 features, and each of them is designed as following:
    // 0: Intercept
    // 1: Positively correlated with the label
    // 2: Negatively correlated with the label
    // 3: Un-correlated with the label
    // 4: Dummy feature 1
    // 5: Dummy feature 2
    // 6-9: Missing features
    val intercept = 1.0
    val variance = 0.001
    val featureIndices = Array(0, 1, 2, 3, 4, 5)
    val features = Array.tabulate(numSamples) { i =>
      val featureValues = Array(
        intercept,
        labels(i) + variance * random.nextGaussian(),
        -labels(i) + variance * random.nextGaussian(),
        random.nextDouble(),
        1.0,
        1.0)
      new SparseVector[Double](featureIndices, featureValues, numFeatures)
    }
    val localDataset =
      LocalDataset(
        Array.tabulate(numSamples)(i => (i.toLong, LabeledPoint(labels(i), features(i), offset = 0.0, weight = 1.0))))

    // don't keep any features
    val filteredDataPoints0 = localDataset.filterFeaturesByPearsonCorrelationScore(numFeaturesToKeep = 0).dataPoints
    assertEquals(filteredDataPoints0.length, numSamples)
    assertTrue(filteredDataPoints0.forall(_._2.features.activeSize == 0))

    // keep 1 feature
    val filteredDataPoints1 = localDataset.filterFeaturesByPearsonCorrelationScore(numFeaturesToKeep = 1).dataPoints
    val filteredDataPointsKeySet1 = filteredDataPoints1.flatMap(_._2.features.activeKeysIterator).toSet
    assertEquals(filteredDataPoints1.length, numSamples)
    assertTrue(filteredDataPoints1.forall(_._2.features.activeSize == 1))
    assertTrue(
      filteredDataPointsKeySet1.size == 1 &&
        (filteredDataPointsKeySet1.contains(0) ||
          filteredDataPointsKeySet1.contains(4) ||
          filteredDataPointsKeySet1.contains(5)),
      s"$filteredDataPointsKeySet1")

    // keep 3 features
    val filteredDataPoints3 = localDataset.filterFeaturesByPearsonCorrelationScore(numFeaturesToKeep = 3).dataPoints
    val filteredDataPointsKeySet3 = filteredDataPoints3.flatMap(_._2.features.activeKeysIterator).toSet
    assertEquals(filteredDataPoints3.length, numSamples)
    assertTrue(filteredDataPoints3.forall(_._2.features.activeSize == 3))
    assertTrue(
      filteredDataPointsKeySet3.size == 3 &&
        filteredDataPointsKeySet3.contains(1) &&
        filteredDataPointsKeySet3.contains(2) &&
        (filteredDataPointsKeySet3.contains(0) ||
          filteredDataPointsKeySet3.contains(4) ||
          filteredDataPointsKeySet3.contains(5)),
      s"$filteredDataPointsKeySet3")

    // keep 5 features
    val filteredDataPoints5 = localDataset.filterFeaturesByPearsonCorrelationScore(numFeaturesToKeep = 5).dataPoints
    val filteredDataPointsKeySet5 = filteredDataPoints5.flatMap(_._2.features.activeKeysIterator).toSet
    assertEquals(filteredDataPoints5.length, numSamples)
    assertTrue(filteredDataPoints5.forall(_._2.features.activeSize == 5))
    assertTrue(filteredDataPointsKeySet5.forall(_ < 6), s"$filteredDataPointsKeySet5")

    // keep all features
    val filteredDataPointsAll = localDataset.filterFeaturesByPearsonCorrelationScore(numFeaturesToKeep = numFeatures)
      .dataPoints
    assertEquals(filteredDataPointsAll.length, numSamples)
    assertTrue(
      filteredDataPointsAll
        .forall(dataPoint => dataPoint._2.features.activeKeysIterator.toSet == Set(0, 1, 2, 3, 4, 5)))
  }
}
