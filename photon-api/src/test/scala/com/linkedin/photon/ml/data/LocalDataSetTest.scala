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
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.constants.MathConst
import com.linkedin.photon.ml.stat.BasicStatisticalSummary
import com.linkedin.photon.ml.test.CommonTestUtils


/**
 * Unit tests for [[LocalDataSet]].
 */
class LocalDataSetTest {

  /**
   * Test the Pearson correlation score computation.
   */
  @Test(groups = Array[String]("testPearsonCorrelationScore", "testCore"))
  def testPearsonCorrelationScore(): Unit = {

    // Test input data
    val labels = Array(1.0, 4.0, 6.0, 9.0)
    val features = Array(
      Vector(0.0, 0.0, 2.0), Vector(5.0, 0.0, -3.0), Vector(7.0, 0.0, -8.0), Vector(0.0, 0.0, -1.0))
    val expected = Map(0 -> 0.05564149, 1 -> 1.0, 2 -> -0.40047142)
    val labelAndFeatures = labels.zip(features)
    val computed = LocalDataSet.computePearsonCorrelationScore(labelAndFeatures)

    computed.foreach { case (key, value) =>
      assertEquals(
        expected(key),
        value,
        CommonTestUtils.LOW_PRECISION_TOLERANCE,
        s"Computed Pearson correlation score is $value, while the expected value is ${expected(key)}.")
    }
  }

  @DataProvider
  def dataForRandomEffectFeatureSelection(): Array[Array[Any]] ={
    val binaryIndices = Set[Int](0)
    val binaryIndicesWithNonBinary = Set[Int](1)
    val nonBinaryIndices = Set[Int]()
    val nonBinaryIndicesWithNonBinary = Set[Int](0)
    // First case x == 0 and y == 0
    val x_0_y_0 = Array(
      SparseVector(1.0),
      SparseVector(1.0),
      SparseVector(1.0),
      SparseVector(1.0),
      SparseVector(1.0)
    )
    val x_0_y_0_l = Array(0.0, 0.0, 0.0, 0.0, 0.0)
    val x_0_y_0_expected = Map(0 -> 0.0)
    val x_0_y_0_stats = BasicStatisticalSummary(
      mean = Vector(0.5),
      variance = Vector(0.0),
      count = 50,
      numNonzeros = Vector(25.0),
      max = Vector(0.0),
      min = Vector(0.0),
      normL1 = Vector(0.0),
      normL2 = Vector(0.0),
      meanAbs = Vector(0.0),
      None: Option[Int]
    )
    val x_0_y_0_g_positive = Array(0.0)

    // Second case x == 0 and y != 0
    val x_0_y_not_0 = Array(
      SparseVector(1.0),
      SparseVector(1.0),
      SparseVector(1.0),
      SparseVector(1.0),
      SparseVector(1.0)
    )
    val x_0_y_not_0_l = Array(0.0, 0.0, 0.0, 0.0, 0.0)

    val x_0_y_not_0_expected = Map(0 -> 0.315959913)

    val x_0_y_not_0_stats = BasicStatisticalSummary(
      mean = Vector(1),
      variance = Vector(0.0),
      count = 50,
      numNonzeros = Vector(50.0),
      max = Vector(0.0),
      min = Vector(0.0),
      normL1 = Vector(0.0),
      normL2 = Vector(0.0),
      meanAbs = Vector(0.0),
      None: Option[Int]
    )
    val x_0_y_not_0_g_positive = Array(50.0)

    // Third case two column, one is non-binary and the other is binary
    val nonBinary = Array(
      SparseVector(3.0, 0.0),
      SparseVector(3.0, 0.0),
      SparseVector(3.0, 0.0),
      SparseVector(3.0, 0.0),
      SparseVector(3.0, 0.0)
    )
    val nonBinary_l = Array(0.0, 0.0, 0.0, 0.0, 0.0)

    val nonBinary_expected = Map(1 -> 0.0)

    val nonBinary_stats = BasicStatisticalSummary(
      mean = Vector(0.5, 0.0),
      variance = Vector(0.0, 0.0),
      count = 50,
      numNonzeros = Vector(25.0, 0.0),
      max = Vector(0.0, 0.0),
      min = Vector(0.0, 0.0),
      normL1 = Vector(0.0, 0.0),
      normL2 = Vector(0.0, 0.0),
      meanAbs = Vector(0.0, 0.0),
      None: Option[Int]
    )
    val nonBinary_g_positive = Array(0.0, 0.0)

    // Forth case x == m and y == n

    val x_m_y_n = Array(
      SparseVector(1.0),
      SparseVector(1.0),
      SparseVector(1.0),
      SparseVector(1.0),
      SparseVector(1.0)
    )
    val x_m_y_n_l = Array(1.0, 1.0, 1.0, 1.0, 1.0)

    val x_m_y_n_expected = Map(0 -> 0.748047336)

    val x_m_y_n_stats = BasicStatisticalSummary(
      mean = Vector(1.0),
      variance = Vector(0.0),
      count = 50,
      numNonzeros = Vector(50.0),
      max = Vector(0.0),
      min = Vector(0.0),
      normL1 = Vector(0.0),
      normL2 = Vector(0.0),
      meanAbs = Vector(0.0),
      None: Option[Int]
    )
    val x_m_y_n_g_positive = Array(50.0)

    // Fifth case t > 1 but not selected

    val t_bt_1_not_selected = Array(
      SparseVector(1.0),
      SparseVector(1.0),
      SparseVector(1.0),
      SparseVector(1.0),
      SparseVector(1.0)
    )
    val t_bt_1_not_selected_l = Array(0.0, 1.0, 1.0, 1.0, 1.0)

    val t_bt_1_not_selected_expected = Map(0 -> 0.96543857)

    val t_bt_1_not_selected_stats = BasicStatisticalSummary(
      mean = Vector(1.0),
      variance = Vector(0.0),
      count = 50,
      numNonzeros = Vector(50.0),
      max = Vector(0.0),
      min = Vector(0.0),
      normL1 = Vector(0.0),
      normL2 = Vector(0.0),
      meanAbs = Vector(0.0),
      None: Option[Int]
    )
    val t_bt_1_not_selected_g_positive = Array(20.0)

    // Sixth case t > 1 selected

    val t_bt_1_selected = Array(
      SparseVector(1.0),
      SparseVector(1.0),
      SparseVector(1.0),
      SparseVector(1.0),
      SparseVector(1.0)
    )
    val t_bt_1_selected_l = Array(1.0, 1.0, 0.0, 0.0, 1.0)

    val t_bt_1_selected_expected = Map(0 -> 1.419596526)

    val t_bt_1_selected_stats = BasicStatisticalSummary(
      mean = Vector(1.0),
      variance = Vector(0.0),
      count = 50,
      numNonzeros = Vector(50.0),
      max = Vector(0.0),
      min = Vector(0.0),
      normL1 = Vector(0.0),
      normL2 = Vector(0.0),
      meanAbs = Vector(0.0),
      None: Option[Int]
    )
    val t_bt_1_selected_g_positive = Array(5.0)

    // Seventh case t < 1 but not selected
    val t_lt_1_not_selected = Array(
      SparseVector(1.0),
      SparseVector(1.0),
      SparseVector(1.0),
      SparseVector(1.0),
      SparseVector(1.0)
    )
    val t_lt_1_not_selected_l = Array(0.0, 0.0, 0.0, 0.0, 1.0)

    val t_lt_1_not_selected_expected = Map(0 -> 0.140309013)

    val t_lt_1_not_selected_stats = BasicStatisticalSummary(
      mean = Vector(1.0),
      variance = Vector(0.0),
      count = 50,
      numNonzeros = Vector(50.0),
      max = Vector(0.0),
      min = Vector(0.0),
      normL1 = Vector(0.0),
      normL2 = Vector(0.0),
      meanAbs = Vector(0.0),
      None: Option[Int]
    )
    val t_lt_1_not_selected_g_positive = Array(15.0)

    // Eighth case t < 1 selected
    val t_lt_1_selected = Array.fill[Vector[Double]](30)(Vector(1.0))

    val t_lt_1_selected_l = Array.fill[Double](30)(0.0)
    Seq(0,1,2).foreach(t_lt_1_selected_l(_) = 1)

    val t_lt_1_selected_expected = Map(0 -> 1.816122265)

    val t_lt_1_selected_stats = BasicStatisticalSummary(
      mean = Vector(1.0),
      variance = Vector(0.0),
      count = 100,
      numNonzeros = Vector(100.0),
      max = Vector(0.0),
      min = Vector(0.0),
      normL1 = Vector(0.0),
      normL2 = Vector(0.0),
      meanAbs = Vector(0.0),
      None: Option[Int]
    )
    val t_lt_1_selected_g_positive = Array(75.0)

    Array(
      Array[Any](x_0_y_0, x_0_y_0_l, x_0_y_0_expected, x_0_y_0_stats, binaryIndices , nonBinaryIndices, x_0_y_0_g_positive, 0),
      Array[Any](x_0_y_not_0, x_0_y_not_0_l, x_0_y_not_0_expected, x_0_y_not_0_stats, binaryIndices, nonBinaryIndices, x_0_y_not_0_g_positive, 0),
      Array[Any](nonBinary, nonBinary_l, nonBinary_expected, nonBinary_stats, binaryIndicesWithNonBinary, nonBinaryIndicesWithNonBinary, nonBinary_g_positive, 1),
      Array[Any](x_m_y_n, x_m_y_n_l, x_m_y_n_expected, x_m_y_n_stats, binaryIndices, nonBinaryIndices, x_m_y_n_g_positive, 0),
      Array[Any](t_bt_1_not_selected, t_bt_1_not_selected_l, t_bt_1_not_selected_expected, t_bt_1_not_selected_stats, binaryIndices, nonBinaryIndices, t_bt_1_not_selected_g_positive, 0),
      Array[Any](t_bt_1_selected, t_bt_1_selected_l, t_bt_1_selected_expected, t_bt_1_selected_stats, binaryIndices, nonBinaryIndices, t_bt_1_selected_g_positive, 1),
      Array[Any](t_lt_1_not_selected, t_lt_1_not_selected_l, t_lt_1_not_selected_expected, t_lt_1_not_selected_stats, binaryIndices, nonBinaryIndices, t_lt_1_not_selected_g_positive, 0),
      Array[Any](t_lt_1_selected, t_lt_1_selected_l, t_lt_1_selected_expected, t_lt_1_selected_stats, binaryIndices, nonBinaryIndices, t_lt_1_selected_g_positive, 1)
    )
  }

  @Test(dependsOnGroups = Array[String]("testComputeRatioCILowerBound", "testCore"),
    dataProvider = "dataForRandomEffectFeatureSelection")
  def testFilterFeaturesByRatioCIBound(
    features: Array[Vector[Double]],
    labels: Array[Double],
    expected: Map[Int, Double],
    globalStats: BasicStatisticalSummary,
    binaryIndices: Set[Int],
    nonbinaryIndices: Set[Int],
    globalPositiveInstances: Array[Double],
    selectedFeatureNum: Int): Unit ={

    val numSamples = labels.length

    val localDataSet =
      LocalDataSet(
        Array.tabulate(numSamples)(i => (i.toLong, LabeledPoint(labels(i), features(i), offset = 0.0, weight = 1.0))))

    val filteredDataPoints = localDataSet
      .filterFeaturesByRatioCIBound(globalStats, globalPositiveInstances, binaryIndices, nonbinaryIndices)
      .dataPoints
    val filteredFeatureNum = filteredDataPoints.map(_._2.features.activeSize)
    assertTrue(filteredFeatureNum.forall( _ == selectedFeatureNum))
  }

  /**
   * Test feature filtering using Pearson correlation score.
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
    val localDataSet =
      LocalDataSet(
        Array.tabulate(numSamples)(i => (i.toLong, LabeledPoint(labels(i), features(i), offset = 0.0, weight = 1.0))))

    // don't keep any features
    val filteredDataPoints0 = localDataSet.filterFeaturesByPearsonCorrelationScore(numFeaturesToKeep = 0).dataPoints
    assertEquals(filteredDataPoints0.length, numSamples)
    assertTrue(filteredDataPoints0.forall(_._2.features.activeSize == 0))

    // keep 1 feature
    val filteredDataPoints1 = localDataSet.filterFeaturesByPearsonCorrelationScore(numFeaturesToKeep = 1).dataPoints
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
    val filteredDataPoints3 = localDataSet.filterFeaturesByPearsonCorrelationScore(numFeaturesToKeep = 3).dataPoints
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
    val filteredDataPoints5 = localDataSet.filterFeaturesByPearsonCorrelationScore(numFeaturesToKeep = 5).dataPoints
    val filteredDataPointsKeySet5 = filteredDataPoints5.flatMap(_._2.features.activeKeysIterator).toSet
    assertEquals(filteredDataPoints5.length, numSamples)
    assertTrue(filteredDataPoints5.forall(_._2.features.activeSize == 5))
    assertTrue(filteredDataPointsKeySet5.forall(_ < 6), s"$filteredDataPointsKeySet5")

    // keep all features
    val filteredDataPointsAll = localDataSet.filterFeaturesByPearsonCorrelationScore(numFeaturesToKeep = numFeatures)
      .dataPoints
    assertEquals(filteredDataPointsAll.length, numSamples)
    assertTrue(
      filteredDataPointsAll
        .forall(dataPoint => dataPoint._2.features.activeKeysIterator.toSet == Set(0, 1, 2, 3, 4, 5)))
  }

  /**
   * Test the Ratio Lower Bound computation
   */
  @Test(groups = Array[String]("testComputeRatioCILowerBound", "testCore"),
    dataProvider = "dataForRandomEffectFeatureSelection")
  def testComputeRatioCILowerBound(
    features: Array[Vector[Double]],
    labels: Array[Double],
    expected: Map[Int, Double],
    globalStats: BasicStatisticalSummary,
    binaryIndices: Set[Int],
    nonBinaryIndices: Set[Int],
    globalPositiveInstances: Array[Double],
    selectedFeatureNum: Int): Unit = {
    // 8 columns of features
    // col 1 for x = 0, y = 0
    // col 2 for x = 0, y != 0
    // col 3 for nonBinary
    // col 4 for x = m, y = n
    // col 5 for t > 1 not selected
    // col 6 for t > 1 selected
    // col 7 for t < 1 not selected
    // col 8 for t < 1 selected

    val labelAndFeatures = labels.zip(features)
    val computed = LocalDataSet.computeRatioCILowerBound(
      labelAndFeatures,
      2.575,
      globalStats,
      globalPositiveInstances,
      binaryIndices)

    computed.foreach { case (key, value) =>
      assertEquals(
        expected(key).asInstanceOf[Double],
        value,
        CommonTestUtils.LOW_PRECISION_TOLERANCE,
        s"Computed Ratio Confidence Interval LowerBound score is $value, while the expected value is ${expected(key)} for key $key." +
          s"the input is features ${features.map{_.toArray}}; labels $labels; globalstat $globalStats; binaryIndices $binaryIndices; globalPositive ${globalPositiveInstances.toArray}")
    }
  }

  /**
   * Test the t and variance computation
   */
  @Test(groups = Array[String]("testComputeMeanAndVariance", "testCore"))
  def testComputeTAndVariance(): Unit = {

    val x = 5.0
    val y = 4.0
    val m = 10.0
    val n = 20.0
    val (t, variance) = LocalDataSet.computeTAndVariance(x, m, y, n)
    assertEquals(t, 2.5, CommonTestUtils.HIGH_PRECISION_TOLERANCE)
    assertEquals(variance, 0.3, CommonTestUtils.HIGH_PRECISION_TOLERANCE)
  }

  /**
   * Test the lowerBound and upperbound computation
   */
  @Test(groups = Array[String]("testLowerBoundAndUpperBound", "testCore"))
  def testUpperBoundAndLowerBound(): Unit = {

    val t = 2.0
    val variance = 4.0
    val quartile = 2.575

    val lowerBound = LocalDataSet.computeLowerBound(t, variance, quartile)
    val upperBound = LocalDataSet.computeUpperBound(t, variance, quartile)
    assertEquals(lowerBound, 0.011598809453684283, CommonTestUtils.HIGH_PRECISION_TOLERANCE)
    assertEquals(upperBound, 344.862980633708, CommonTestUtils.HIGH_PRECISION_TOLERANCE)
  }
}
