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
package com.linkedin.photon.ml.test

import breeze.linalg.{SparseVector, Vector}
import org.apache.commons.math3.distribution.PascalDistribution
import org.apache.commons.math3.random.{RandomGenerator, Well19937a}
import org.apache.log4j.{LogManager, Logger}
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.immutable.TreeMap
import scala.collection.mutable.ListBuffer

// TODO: Additional documentation required
trait SparkTestUtils {

  var sc: SparkContext = _

  /**
   * Provides a synchronized block for methods to safely create their own Spark contexts without stomping on others.
   * Users are expected to handle Spark context creation and cleanup correctly.
   *
   * @param name the test job name
   * @param body the execution closure
   */
  def sparkTestSelfServeContext(name: String)(body: => Unit): Unit = {
    SparkTestUtils.SPARK_LOCAL_CONFIG.synchronized {
      try {
        body
      } finally {
        System.clearProperty("spark.driver.port")
        System.clearProperty("spark.hostPort")
      }
    }
  }

  /**
   * Provides a synchronized block with an auto-created safe Spark context. This wrapper will handle both creation and
   * cleanup of the context.
   *
   * @param name the test job name
   * @param body the execution closure
   */
  def sparkTest(name: String)(body: => Unit): Unit = {
    SparkTestUtils.SPARK_LOCAL_CONFIG.synchronized {

      val conf: SparkConf = new SparkConf()
      conf.setAppName(name).setMaster(SparkTestUtils.SPARK_LOCAL_CONFIG)
      sc = new SparkContext(conf)

      try {
        body
      } finally {
        sc.stop()
        System.clearProperty("spark.driver.port")
        System.clearProperty("spark.hostPort")
      }
    }
  }

  def drawBalancedSampleFromNumericallyBenignDenseFeaturesForBinaryClassifierLocal(
      seed: Int,
      size: Int,
      dimensionality: Int): Iterator[(Double, Vector[Double])] = {

    val desiredSparsity = 0.1
    val positiveLabel = 1
    val negativeLabel = 0
    val probabilityPositive = 0.5

    SparkTestUtils.numericallyBenignGeneratorFunctionForBinaryClassifier(
      seed, desiredSparsity, dimensionality, positiveLabel, negativeLabel, probabilityPositive, 0,
      (0 until size).iterator)
  }

  def drawSampleFromNumericallyBenignDenseFeaturesForPoissonRegressionLocal(
      seed: Int,
      size: Int,
      dimensionality: Int) : Iterator[(Double, Vector[Double])] =  {

    val desiredSparsity = 0.1
    SparkTestUtils.numericallyBenignGeneratorFunctionForPoissonRegression(
      seed, desiredSparsity, dimensionality, 0, (0 to size).iterator)
  }

  def drawSampleFromNumericallyBenignDenseFeaturesForLinearRegressionLocal(
      seed: Int,
      size: Int,
      dimensionality: Int) : Iterator[(Double, Vector[Double])] =  {

    val desiredSparsity = 0.1
    SparkTestUtils.numericallyBenignGeneratorFunctionForLinearRegression(
      seed, desiredSparsity, dimensionality, 0, (0 to size).iterator)
  }

  def drawBalancedSampleFromOutlierDenseFeaturesForBinaryClassifierLocal(
      seed: Int,
      size: Int,
      dimensionality: Int) : Iterator[(Double, Vector[Double])] =  {

    val desiredSparsity = 0.1
    val positiveLabel = 1
    val negativeLabel = 0
    val probabilityPositive = 0.5

    SparkTestUtils.outlierGeneratorFunctionForBinaryClassifier(
      seed, desiredSparsity, dimensionality, positiveLabel, negativeLabel, probabilityPositive, 0,
      (0 to size).iterator)
  }

  def drawSampleFromOutlierDenseFeaturesForPoissonRegressionLocal(
      seed: Int,
      size: Int,
      dimensionality: Int) : Iterator[(Double, Vector[Double])] = {

    val desiredSparsity = 0.1
    SparkTestUtils.outlierGeneratorFunctionForPoissonRegression(
      seed, desiredSparsity, dimensionality, 0, (0 to size).iterator)
  }

  def drawSampleFromOutlierDenseFeaturesForLinearRegressionLocal(
      seed: Int,
      size: Int,
      dimensionality: Int) : Iterator[(Double, Vector[Double])] = {

    val desiredSparsity = 0.1
    SparkTestUtils.outlierGeneratorFunctionForLinearRegression(
      seed, desiredSparsity, dimensionality, 0, (0 to size).iterator)
  }

  def drawBalancedSampleFromInvalidDenseFeaturesForBinaryClassifierLocal(
      seed: Int,
      size: Int,
      dimensionality: Int) : Iterator[(Double, Vector[Double])] = {

    val desiredSparsity = 0.1
    val positiveLabel = 1
    val negativeLabel = 0
    val probabilityPositive = 0.5

    SparkTestUtils.invalidFeatureGeneratorFunctionForBinaryClassifier(
      seed, desiredSparsity, dimensionality, positiveLabel, negativeLabel, probabilityPositive, 0,
      (0 until size).iterator)
  }

  def drawSampleFromInvalidDenseFeaturesForPoissonRegressionLocal(
      seed: Int,
      size: Int,
      dimensionality: Int) : Iterator[(Double, Vector[Double])] =  {

    val desiredSparsity = 0.1
    SparkTestUtils.invalidFeatureGeneratorFunctionForPoissonRegression(
      seed, desiredSparsity, dimensionality, 0, (0 to size).iterator)
  }

  def drawSampleFromInvalidDenseFeaturesForLinearRegressionLocal(
      seed: Int,
      size: Int,
      dimensionality: Int) : Iterator[(Double, Vector[Double])] =  {

    val desiredSparsity = 0.1
    SparkTestUtils.invalidFeatureGeneratorFunctionForLinearRegression(
      seed, desiredSparsity, dimensionality, 0, (0 to size).iterator)
  }

  def drawSampleFromInvalidLabels(
      seed:Int,
      size:Int,
      dimensionality:Int) : Iterator[(Double, Vector[Double])] = {

    val desiredSparsity = 0.1
    SparkTestUtils.invalidLabelGeneratorFunction(seed, desiredSparsity, dimensionality, 0, (0 to size).iterator)
  }
}

object SparkTestUtils {
  val log: Logger = LogManager.getLogger(classOf[SparkTestUtils])

  val SPARK_LOCAL_CONFIG: String = "local[4]"
  val INLIER_PROBABILITY: Double = 0.90
  val INLIER_STANDARD_DEVIATION: Double = 1e-3
  val OUTLIER_STANDARD_DEVIATION: Double = 1

  def numericallyBenignGeneratorFunctionForBinaryClassifier(
      seed: Int,
      desiredSparsity: Double,
      desiredDimensionality: Int,
      positiveLabel: Int,
      negativeLabel: Int,
      probabilityPositive: Double,
      index: Int,
      items: Iterator[Int]) : Iterator[(Double, Vector[Double])] = {

    val prng:RandomGenerator = new Well19937a(5000 * seed + index)
    val negBinomial:PascalDistribution = new PascalDistribution(prng, 1, desiredSparsity)
    val result: ListBuffer[(Double, Vector[Double])] = ListBuffer()

    while (items.hasNext) {
      items.next()

      // Assign a class
      val classLabel: Int = if (prng.nextDouble <= probabilityPositive) {
        positiveLabel
      } else {
        negativeLabel
      }

      // strictly separable ( x >= 1 --> positive, x <= -1 --> negative, no samples in the middle)
      val tmpXAttribute: Double = 0.1 + 0.9 * prng.nextDouble()
      val xAttribute: Double = if (classLabel == positiveLabel) {
        tmpXAttribute
      } else {
        -tmpXAttribute
      }

      result += generateNumericallyBenignSparseVector(
        classLabel.toDouble, xAttribute, desiredDimensionality, prng, negBinomial)
    }
    result.toList.iterator
  }

  def outlierGeneratorFunctionForBinaryClassifier(
      seed: Int,
      desiredSparsity: Double,
      desiredDimensionality: Int,
      positiveLabel: Int,
      negativeLabel: Int,
      probabilityPositive: Double,
      index: Int,
      items: Iterator[Int]) : Iterator[(Double, Vector[Double])] = {

    val prng:RandomGenerator = new Well19937a(5000 * seed + index)
    val negBinomial:PascalDistribution = new PascalDistribution(prng, 1, desiredSparsity)
    val result: ListBuffer[(Double, Vector[Double])] = ListBuffer()

    while (items.hasNext) {
      items.next()

      // Assign a class
      val classLabel: Int = if (prng.nextDouble <= probabilityPositive) {
        positiveLabel
      } else {
        negativeLabel
      }

      // strictly separable ( x >= 1 --> positive, x <= -1 --> negative, no samples in the middle)
      val tmpXAttribute: Double = 0.1 + 0.9 * prng.nextDouble()
      val xAttribute: Double = if (classLabel == positiveLabel) {
        tmpXAttribute
      } else {
        -tmpXAttribute
      }

      result += generateSparseVectorWithOutliers(
        classLabel.toDouble, xAttribute, desiredDimensionality, prng, negBinomial)
    }
    result.toList.iterator
  }

  def invalidFeatureGeneratorFunctionForBinaryClassifier(
      seed: Int,
      desiredSparsity: Double,
      desiredDimensionality: Int,
      positiveLabel: Int,
      negativeLabel: Int,
      probabilityPositive: Double,
      index: Int,
      items: Iterator[Int]) : Iterator[(Double, Vector[Double])] = {

    val prng:RandomGenerator = new Well19937a(5000 * seed + index)
    val negBinomial:PascalDistribution = new PascalDistribution(prng, 1, desiredSparsity)
    val result: ListBuffer[(Double, Vector[Double])] = ListBuffer()

    while (items.hasNext) {
      items.next()

      // Assign a class
      val classLabel: Int = if (prng.nextDouble <= probabilityPositive) {
        positiveLabel
      } else {
        negativeLabel
      }

      // strictly separable ( x >= 1 --> positive, x <= -1 --> negative, no samples in the middle)
      val tmpXAttribute: Double = 0.1 + 0.9 * prng.nextDouble()
      val xAttribute: Double = if (classLabel == positiveLabel) {
        tmpXAttribute
      } else {
        -tmpXAttribute
      }

      result += generateSparseVectorWithInvalidValues(
        classLabel.toDouble, xAttribute, desiredDimensionality, prng, negBinomial)
    }
    result.toList.iterator
  }

  def numericallyBenignGeneratorFunctionForPoissonRegression(
      seed: Int,
      desiredSparsity: Double,
      desiredDimensionality: Int,
      index: Int,
      items: Iterator[Int]) : Iterator[(Double, Vector[Double])] = {

    val prng:RandomGenerator = new Well19937a(5000 * seed + index)
    val negBinomial:PascalDistribution = new PascalDistribution(prng, 1, desiredSparsity)
    val result: ListBuffer[(Double, Vector[Double])] = ListBuffer()

    while (items.hasNext) {
      items.next()

      // Assign a class
      val classLabel = 1 + prng.nextDouble() * 10

      // make sure we account for scaling
      val xAttribute = (Math.log(classLabel) + prng.nextGaussian() * INLIER_STANDARD_DEVIATION)/Math.log(11.0)

      result += generateNumericallyBenignSparseVector(
        classLabel, xAttribute, desiredDimensionality, prng, negBinomial)
    }

    result.toList.iterator
  }

  def outlierGeneratorFunctionForPoissonRegression(
      seed: Int,
      desiredSparsity: Double,
      desiredDimensionality: Int,
      index: Int,
      items: Iterator[Int]) : Iterator[(Double, Vector[Double])] = {

    val prng:RandomGenerator = new Well19937a(5000 * seed + index)
    val negBinomial:PascalDistribution = new PascalDistribution(prng, 1, desiredSparsity)
    val result: ListBuffer[(Double, Vector[Double])] = ListBuffer()

    while (items.hasNext) {
      items.next()

      // Assign a class
      val classLabel = 1 + prng.nextDouble() * 10

      // make sure we account for scaling
      val xAttribute = (Math.log(classLabel) + prng.nextGaussian() * INLIER_STANDARD_DEVIATION)/Math.log(11.0)

      result += generateSparseVectorWithOutliers(
        classLabel, xAttribute, desiredDimensionality, prng, negBinomial)
    }

    result.toList.iterator
  }

  def invalidFeatureGeneratorFunctionForPoissonRegression(
      seed: Int,
      desiredSparsity: Double,
      desiredDimensionality: Int,
      index: Int,
      items: Iterator[Int]) : Iterator[(Double, Vector[Double])] = {

    val prng:RandomGenerator = new Well19937a(5000 * seed + index)
    val negBinomial:PascalDistribution = new PascalDistribution(prng, 1, desiredSparsity)
    val result: ListBuffer[(Double, Vector[Double])] = ListBuffer()

    while (items.hasNext) {
      items.next()

      // Assign a class
      val classLabel = 1 + prng.nextDouble() * 10

      // make sure we account for scaling
      val xAttribute = (Math.log(classLabel) + prng.nextGaussian() * INLIER_STANDARD_DEVIATION)/Math.log(11.0)

      result += generateSparseVectorWithInvalidValues(
        classLabel, xAttribute, desiredDimensionality, prng, negBinomial)
    }

    result.toList.iterator
  }

  def numericallyBenignGeneratorFunctionForLinearRegression(
      seed: Int,
      desiredSparsity: Double,
      desiredDimensionality: Int,
      index: Int,
      items: Iterator[Int]) : Iterator[(Double, Vector[Double])] = {

    val prng:RandomGenerator = new Well19937a(5000 * seed + index)
    val negBinomial:PascalDistribution = new PascalDistribution(prng, 1, desiredSparsity)
    val result: ListBuffer[(Double, Vector[Double])] = ListBuffer()

    while (items.hasNext) {
      items.next()

      // Assign a label
      val classLabel = 2*prng.nextDouble() - 1
      val xAttribute = classLabel + prng.nextGaussian() * INLIER_STANDARD_DEVIATION
      result += generateNumericallyBenignSparseVector(
        classLabel, xAttribute, desiredDimensionality, prng, negBinomial)
    }

    result.toList.iterator
  }

  def outlierGeneratorFunctionForLinearRegression(
      seed: Int,
      desiredSparsity: Double,
      desiredDimensionality: Int,
      index: Int,
      items: Iterator[Int]) : Iterator[(Double, Vector[Double])] = {

    val prng:RandomGenerator = new Well19937a(5000 * seed + index)
    val negBinomial:PascalDistribution = new PascalDistribution(prng, 1, desiredSparsity)
    val result: ListBuffer[(Double, Vector[Double])] = ListBuffer()

    while (items.hasNext) {
      items.next()

      // Assign a label
      val classLabel = 1 + prng.nextDouble()
      val xAttribute = classLabel - 1 + prng.nextGaussian() * OUTLIER_STANDARD_DEVIATION
      result += generateSparseVectorWithOutliers(
        classLabel, xAttribute, desiredDimensionality, prng, negBinomial)
    }

    result.toList.iterator
  }

  def invalidFeatureGeneratorFunctionForLinearRegression(
      seed: Int,
      desiredSparsity: Double,
      desiredDimensionality: Int,
      index: Int,
      items: Iterator[Int]) : Iterator[(Double, Vector[Double])] = {

    val prng:RandomGenerator = new Well19937a(5000 * seed + index)
    val negBinomial:PascalDistribution = new PascalDistribution(prng, 1, desiredSparsity)
    val result: ListBuffer[(Double, Vector[Double])] = ListBuffer()

    while (items.hasNext) {
      items.next()

      // Assign a label
      val classLabel = 1 + prng.nextDouble()
      val xAttribute = classLabel - 1 + prng.nextGaussian() * OUTLIER_STANDARD_DEVIATION
      result += generateSparseVectorWithInvalidValues(
        classLabel, xAttribute, desiredDimensionality, prng, negBinomial)
    }

    result.toList.iterator
  }

  def invalidLabelGeneratorFunction(
      seed: Int,
      desiredSparsity: Double,
      desiredDimensionality: Int,
      index: Int,
      items: Iterator[Int]) : Iterator[(Double, Vector[Double])] = {

    val prng:RandomGenerator = new Well19937a(5000 * seed + index)
    val negBinomial:PascalDistribution = new PascalDistribution(prng, 1, desiredSparsity)
    val result: ListBuffer[(Double, Vector[Double])] = ListBuffer()

    while (items.hasNext) {
      items.next()

      // Assign a class
      val classLabel: Double = prng.nextInt(3) match {
        case 0 => Double.PositiveInfinity
        case 1 => Double.NegativeInfinity
        case _ => Double.NaN
      }

      // strictly separable ( x >= 1 --> positive, x <= -1 --> negative, no samples in the middle)
      val xAttribute: Double = 0.1 + 0.9 * prng.nextDouble()

      result += generateNumericallyBenignSparseVector(
        classLabel.toDouble, xAttribute, desiredDimensionality, prng, negBinomial)
    }

    result.toList.iterator
  }

  // In this case, numerically benign means that all values are pretty uniformly distributed between
  // -1 and 1
  def generateNumericallyBenignSparseVector(
      label: Double,
      xAttribute: Double,
      desiredDimensionality: Int,
      prng: RandomGenerator,
      negBinomial: PascalDistribution) : (Double, Vector[Double]) = {

    // temporary storage for our items
    var features: Map[Int, Double] = TreeMap()

    features += (0 -> xAttribute)

    // Fill in the rest with dummy binary variables
    //
    // Trick: instead of generating exactly desiredDimensionality calls to nextDouble() to
    // simulate a coin toss for each element in the vector, instead generate a sample from
    // NB(1, desiredSparsity) which tells us how far to skip ahead to the next successful
    // coin toss. This is *WAY* faster
    var i: Int = 1 + negBinomial.sample()

    while (i < desiredDimensionality) {
      features += (i -> 2 * (prng.nextDouble() - 0.5))
      i += negBinomial.sample()
    }

    // Turn our temporary structures into our output sample
    val indices = new Array[Int](features.size)
    val values = new Array[Double](features.size)
    i = 0

    for ((idx: Int, v: Double) <- features) {
      indices(i) = idx
      values(i) = v
      i += 1
    }

    (label, new SparseVector[Double](indices, values, indices.length, desiredDimensionality))
  }

  def generateSparseVectorWithOutliers(
      label: Double,
      xAttribute: Double,
      desiredDimensionality: Int,
      prng: RandomGenerator,
      negBinomial: PascalDistribution) : (Double, Vector[Double]) = {

    // temporary storage for our items
    var features: Map[Int, Double] = TreeMap()

    features += (0 -> xAttribute)

    // Fill in the rest with dummy binary variables
    //
    // Trick: instead of generating exactly desiredDimensionality calls to nextDouble() to
    // simulate a coin toss for each element in the vector, instead generate a sample from
    // NB(1, desiredSparsity) which tells us how far to skip ahead to the next successful
    // coin toss. This is *WAY* faster
    {
      var i: Int = 1 + negBinomial.sample()

      while (i < desiredDimensionality) {
        // With probability > 0.99, generate a gaussian sample with zero mean and tiny covariance,
        // with probability 0.01, generate +-1
        if (prng.nextDouble() < INLIER_PROBABILITY) {
          features += (i -> prng.nextGaussian() * INLIER_STANDARD_DEVIATION)
        } else if (prng.nextBoolean()) {
          features += (i -> 1)
        } else {
          features += (i -> -1)
        }

        i += negBinomial.sample()
      }
    }

    // Turn our temporary structures into our output sample
    val indices = new Array[Int](features.size)
    val values = new Array[Double](features.size)
    var i: Int = 0

    for ((idx: Int, v: Double) <- features) {
      indices(i) = idx
      values(i) = v
      i += 1
    }

    (label, new SparseVector[Double](indices, values, indices.length, desiredDimensionality))
  }

  def generateSparseVectorWithInvalidValues(
      label: Double,
      xAttribute: Double,
      desiredDimensionality: Int,
      prng: RandomGenerator,
      negBinomial: PascalDistribution) : (Double, Vector[Double]) = {

    // temporary storage for our items
    var features: Map[Int, Double] = TreeMap()
    features += (0 -> xAttribute)

    // Fill in the rest with dummy binary variables
    //
    // Trick: instead of generating exactly desiredDimensionality calls to nextDouble() to
    // simulate a coin toss for each element in the vector, instead generate a sample from
    // NB(1, desiredSparsity) which tells us how far to skip ahead to the next successful
    // coin toss. This is *WAY* faster
    var i: Int = 1 + negBinomial.sample()

    while (i < desiredDimensionality - 3) {
      // With probability > 0.99, generate a gaussian sample with zero mean and tiny covariance,
      // with probability 0.01, generate +-1
      if (prng.nextDouble() < INLIER_PROBABILITY) {
        features += (i -> prng.nextGaussian() * INLIER_STANDARD_DEVIATION)
      } else {
        prng.nextInt(3) match {
          case 0 => features += (i -> Double.NaN)
          case 1 => features += (i -> Double.PositiveInfinity)
          case 2 => features += (i -> Double.NegativeInfinity)
        }
      }

      i += negBinomial.sample()
    }
    features += (desiredDimensionality - 3 -> Double.NaN)
    features += (desiredDimensionality - 2 -> Double.PositiveInfinity)
    features += (desiredDimensionality - 1 -> Double.NegativeInfinity)

    // Turn our temporary structures into our output sample
    val indices = new Array[Int](features.size)
    val values = new Array[Double](features.size)
    var j: Int = 0

    for ((idx: Int, v: Double) <- features) {
      indices(j) = idx
      values(j) = v
      j += 1
    }

    (label, new SparseVector[Double](indices, values, indices.length, desiredDimensionality))
  }
}
