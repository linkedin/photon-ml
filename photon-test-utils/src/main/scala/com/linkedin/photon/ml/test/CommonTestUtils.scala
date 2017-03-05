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
package com.linkedin.photon.ml.test

import scala.collection.mutable.ListBuffer
import scala.util.Random

import breeze.linalg.{DenseVector, Vector}

/**
 * A collection of methods useful for tests.
 */
object CommonTestUtils {

  /**
   * Append prefix to a CMD line option name, forming an argument string.
   *
   * @param optionName The option name
   * @return The argument string
   */
  def fromOptionNameToArg(optionName: String): String = "--" + optionName

  /**
   * This a utility for comparing two constraint maps in unit tests. Returns true if both the options are None or are
   * non-empty and the contained maps have the same set of (key, value) pairs.
   *
   * @param m1 Option containing the first map
   * @param m2 Option containing the second map
   * @return True iff both options are none or contain maps that have the exact same set of (key, value) tuples
   */
  def compareConstraintMaps(m1: Option[Map[Int, (Double, Double)]], m2: Option[Map[Int, (Double, Double)]]): Boolean = {
    (m1, m2) match {
      case (Some(x), Some(y)) =>
        if (x.size != y.size) {
          false
        } else {
          x.count {
            case (id, bounds) =>
              y.get(id) match {
                case Some(w: (Double, Double)) => w == bounds
                case None => false
              }
          } == x.size
        }
      case (None, None) => true
      case (None, _) => false
      case (_, None) => false
    }
  }

  /**
    * Samples a dense vector from a Gaussian with the given properties.
    *
    * @param dim The dimension of the vector
    * @param mean The mean of the distribution
    * @param sd The standard deviation of the distribution
    * @param seed The random seed value (defaults to current system time)
    * @return A dense vector with values sampled from the Gaussian
    */
  def generateDenseVector(
      dim: Int,
      mean: Double = 0,
      sd: Double = 1,
      seed: Long = System.currentTimeMillis): DenseVector[Double] = {

    val random = new Random(seed)
    DenseVector(Seq.fill(dim)({ (random.nextGaussian + mean) * sd }).toArray)
  }

  /**
   * Generates given number of valid and invalid dense feature vectors of given dimension.
   *
   * @param numValidVectors Number of valid vectors to generate
   * @param numInvalidVectors Number of invalid vectors to generate
   * @param numDimensions The dimension of the generated feature vectors
   * @return A sequence of generated dense feature vectors
   */
  def generateDenseFeatureVectors(
      numValidVectors: Int, numInvalidVectors: Int,
      numDimensions: Int): Seq[Vector[Double]] = {

    val r = new Random(System.currentTimeMillis())
    val result = new ListBuffer[Vector[Double]]

    for (_ <- 0 until numValidVectors) {
      val v = new Array[Double](numDimensions)
      for (j <- 0 until numDimensions) {
        v(j) = r.nextDouble()
      }
      result +=  new DenseVector[Double](v)
    }

    for (_ <- 0 until numInvalidVectors) {
      val v = new Array[Double](numDimensions)
      /* In the worst case, if all our coin tosses below fail, we might end up with a valid vector.
         Hence setting the first element invalid to guarantee an invalid vector */
      v(0) = Double.NaN
      for (j <- 1 until numDimensions) {
        v(j) = if (r.nextBoolean()) {
          r.nextInt(3) match {
            case 0 => Double.NaN
            case 1 => Double.PositiveInfinity
            case 2 => Double.NegativeInfinity
          }
        } else {
          r.nextDouble()
        }
      }
      result += new DenseVector[Double](v)
    }

    result
  }

  /**
   * Convert the option -> value map into an argument array.
   *
   * @param map Map of option to option setting
   * @return Array representation of arguments
   */
  def argArray(map: Map[String, String]): Array[String] = map.foldLeft(Array[String]()) {
    case (array, (option, value)) =>
      array :+ CommonTestUtils.fromOptionNameToArg(option) :+ value
  }

  /**
   * Convert a [[Map]] of option name and value into a [[Seq]] of arguments.
   */
  def mapToArray(args: Map[String, String]): Array[String] =
    args.toArray.flatMap { case (name, value) => Seq(name, value) }

  /**
   * Create tuples of score, label, and weight by pairing two arrays of scores and labels, then adding a default weight.
   *
   * @param scores An array of scores
   * @param labels An array of labels
   * @param weight A default weight
   * @return An array of (score, label, weight) tuples
   */
  def getScoreLabelAndWeights(
      scores: Array[Double],
      labels: Array[Double],
      weight: Double = 1.0): Array[(Double, Double, Double)] =
    scores.zip(labels).map { case (score, label) => (score, label, weight) }

  /**
   * Pair each element in an array with an index based on starting index and its position in the array.
   *
   * @param arr Input array of elements
   * @param startIndex Index from which to begin
   * @tparam T Type of array input element
   * @return The original elements of the array paired an index, in ascending order
   */
  def zipWithIndex[T](arr: Iterable[T], startIndex: Int = 0): Array[(Long, T)] =
    arr.zipWithIndex.map { case (t, idx) => ((idx + startIndex).toLong, t) }.toArray
}
