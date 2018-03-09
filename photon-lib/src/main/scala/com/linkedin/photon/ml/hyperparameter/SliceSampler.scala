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
package com.linkedin.photon.ml.hyperparameter

import scala.annotation.tailrec
import scala.util.Random

import breeze.linalg.{DenseVector, norm}
import breeze.numerics.log
import breeze.stats.distributions.Gaussian

/**
 * This class implements the slice sampling algorithm. Slice sampling is an Markov chain Monte Carlo algorithm for
 * sampling from an arbitrary continuous distribution function.
 *
 * Consider the ASCII-normal distribution function shown below.
 *
 *
 *                  /-\                /-\ f(x)           /-\                /-\                /-\
 *                 /   \              /  |\              /  |\              /   \              /   \
 *                /     \            /   | \            /---|-\            /-x'--\            /     \
 *             --/    x  \--      --/    x  \--      --/    x  \--      --/       \--      --/  x'   \--
 *
 *                   1                  2                  3                  4                  5
 *
 * The procedure for drawing a sample from the function is as follows:
 *
 *   1) Start with an initial point x in the domain (e.g. the sample from a previous iteration).
 *   2) Sample a vertical value uniformly between zero and f(x).
 *   3) Step out horizontally in both directions from the value obtained in step 2 until reaching the boundary of the
          function. This is the "slice".
 *   4) Sample uniformly from the slice to obtain the new point x'.
 *   5) Back to step 1 with the new point.
 *
 * @param stepSize the size to expand a slice while taking one step in a single direction from the
 *   starting point
 * @param maxStepsOut the maximum number of steps in either direction while expanding a slice
 * @param seed the random seed
 */
class SliceSampler(
    stepSize: Double = 1.0,
    maxStepsOut: Int = 1000,
    seed: Long = System.currentTimeMillis) {

  val random = new Random(seed)

  /**
   * Draws a new point from the target distribution
   *
   * @param x the original point
   * @param logp the log-transformed function from which to sample. Note that this need not sum to 1, and need only be
   *   proportional to the target PDF.
   * @return the new point
   */
  def draw(x: DenseVector[Double], logp: (DenseVector[Double]) => Double): DenseVector[Double] = {
    val sampledVector = DenseVector(Gaussian(0, 1).sample(x.length):_*)
    val direction = sampledVector / norm(sampledVector)

    draw(x, logp, direction)
  }

  /**
   * Draws a new point from the target distribution, sampling from each dimension one-by-one
   *
   * @param x the original point
   * @param logp the log-transformed function from which to sample. Note that this need not sum to 1, and need only be
   *   proportional to the target PDF.
   * @return the new point
   */
  def drawDimensionWise(x: DenseVector[Double], logp: (DenseVector[Double]) => Double): DenseVector[Double] = {
    val directions = random.shuffle((0 until x.length).toList)
    directions.foldLeft(x) { (currX, i) =>
      draw(currX, logp, getDirection(i, x.length))
    }
  }

  /**
   * Draws a new point from the target distribution along the given direction
   *
   * @param x the original point
   * @param logp the log-transformed function from which to sample. Note that this need not sum to 1, and need only be
   *   proportional to the target PDF.
   * @param direction the direction along which to draw a new point
   * @return the new point
   */
  protected def draw(
      x: DenseVector[Double],
      logp: (DenseVector[Double]) => Double,
      direction: DenseVector[Double]): DenseVector[Double] = {

    val y = log(random.nextDouble()) + logp(x)
    val slice = stepOut(x, y, logp, direction)

    draw(x, y, logp, direction, slice)
  }

  /**
   * Draws a new point uniformly from the given horizontal slice through the function, along the
   * given direction
   *
   * @param x the original point
   * @param y the value of the function slice
   * @param logp the log-transformed function from which to sample. Note that this need not sum to 1, and need only be
   *   proportional to the target PDF.
   * @param direction the direction along which to draw a new point
   * @param slice the slice bounds
   * @return the new point
   */
  @tailrec
  protected final def draw(
      x: DenseVector[Double],
      y: Double,
      logp: (DenseVector[Double]) => Double,
      direction: DenseVector[Double],
      slice: (DenseVector[Double], DenseVector[Double])): DenseVector[Double] = {

    // Sample uniformly from the slice
    val (lower, upper) = slice
    val newX = lower + random.nextDouble() * (upper - lower)

    if (logp(newX) > y) {
      // If we've landed in at point of the PDF that's above the slice, return the point
      newX

    } else {
      // Otherwise, reject the sample and shrink the slice
      val newSlice = shrink(slice, x, newX, direction)
      draw(x, y, logp, direction, newSlice)
    }
  }

  /**
   * Builds a direction basis vector from the index
   *
   * @param i the active index
   * @param dim size of the vector
   * @return the direction basis vector
   */
  protected def getDirection(i: Int, dim: Int) =
    DenseVector.tabulate(dim) { j => if (i == j) 1.0 else 0.0 }

  /**
   * Performs the step out procedure. Start at the given x position, and expand outwards along the
   * given direction until the slice extends beyond where the PDF lies above the y value.
   *
   * @param x the starting point
   * @param y the value at which the function will be sliced
   * @param logp the log-transformed function from which to sample. Note that this need not sum to 1, and need only be
   *   proportional to the target PDF.
   * @param direction the direction along which to slice
   * @return the slice lower and upper bounds
   */
  protected def stepOut(
      x: DenseVector[Double],
      y: Double,
      logp: (DenseVector[Double]) => Double,
      direction: DenseVector[Double]): (DenseVector[Double], DenseVector[Double]) = {

    var lower = x - direction * random.nextDouble() * stepSize
    var upper = lower + direction * stepSize
    var lowerStepsOut = 0
    var upperStepsOut = 0

    while ((logp(lower) > y) && lowerStepsOut < maxStepsOut) {
      lower -= direction * stepSize
      lowerStepsOut += 1
    }

    while ((logp(upper) > y) && upperStepsOut < maxStepsOut) {
      upper += direction * stepSize
      upperStepsOut += 1
    }

    (lower, upper)
  }

  /**
   * Shrinks the slice by clamping to the new point x.
   *
   * @param slice the original slice
   * @param x the original point x
   * @param newX the new point x
   * @param direction the direction along which to shrink the slice
   * @return the shrunken slice
   */
  protected def shrink(
      slice: (DenseVector[Double], DenseVector[Double]),
      x: DenseVector[Double],
      newX: DenseVector[Double],
      direction: DenseVector[Double]): (DenseVector[Double], DenseVector[Double]) = {

    val (lower, upper) = slice

    if ((newX.t * direction) < (x.t * direction)) {
      (newX, upper)

    } else if ((newX.t * direction) > (x.t * direction)) {
      (lower, newX)

    } else {
      throw new RuntimeException("Slice size shrank to zero.")
    }
  }
}
