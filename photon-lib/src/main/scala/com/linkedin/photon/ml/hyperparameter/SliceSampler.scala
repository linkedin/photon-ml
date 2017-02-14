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

import breeze.linalg.DenseVector
import breeze.numerics.log

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
 * @param logp the log-transformed function from which to sample. Note that this need not sum to 1, and need only be
 *   proportional to the target PDF.
 * @param range the range of values within which to sample
 * @param stepSize the size to expand a slice while taking one step in a single direction from the
 *   starting point
 * @param seed the random seed
 */
class SliceSampler(
    logp: (DenseVector[Double]) => Double,
    range: (Double, Double) = (log(1e-5), log(1e5)),
    stepSize: Double = 1.0,
    seed: Long = System.currentTimeMillis) {

  val random = new Random(seed)

  /**
   * Draws a new point from the target distribution
   *
   * @param x the original point
   * @return the new point
   */
  def draw(x: DenseVector[Double]): DenseVector[Double] = {
    val directions = random.shuffle((0 until x.length).toList)
    directions.foldLeft(x) { (currX, i) =>
      draw(currX, getDirection(i, x.length))
    }
  }

  /**
   * Draws a new point from the target distribution along the given direction
   *
   * @param x the original point
   * @param direction the direction along which to draw a new point
   * @return the new point
   */
  protected def draw(
      x: DenseVector[Double],
      direction: DenseVector[Double]): DenseVector[Double] = {

    require(checkDirection(direction), "Direction must be a standard unit basis vector.")

    val y = log(random.nextDouble()) + logp(x)
    val slice = stepOut(x, y, direction)

    draw(x, y, direction, slice)
  }

  /**
   * Draws a new point uniformly from the given horizontal slice through the function, along the
   * given direction
   *
   * @param x the original point
   * @param y the value of the function slice
   * @param direction the direction along which to draw a new point
   * @param slice the slice bounds
   * @return the new point
   */
  @tailrec
  protected final def draw(
      x: DenseVector[Double],
      y: Double,
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
      val newSlice = try {
        shrink(slice, x, newX, direction)

      } catch {
        // If shrinking the slice fails, it means that our step-out procedure failed to distinguish between regions of
        // the PDF that fall above or below the slice, and our slice shrank to zero. In this case, the best we can do is
        // reset to the entire range and start over with rejection sampling and shrinking.
        case _: Exception => (range._1 * direction, range._2 * direction)
      }

      draw(x, y, direction, newSlice)
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
   * Verifies that the direction vector is a standard unit basis vector
   *
   * @param direction the direction vector to check
   * @return true if valid
   */
  protected def checkDirection(direction: DenseVector[Double]): Boolean = {
    val arr = direction.toArray
    arr.count(_ == 1.0) == 1 &&
      arr.count(_ == 0.0) == direction.length - 1
  }

  /**
   * Performs the step out procedure. Start at the given x position, and expand outwards along the
   * given direction until the slice extends beyond where the PDF lies above the y value.
   *
   * @param x the starting point
   * @param y the value at which the function will be sliced
   * @param direction the direction along which to slice
   * @return the slice lower and upper bounds
   */
  protected def stepOut(
      x: DenseVector[Double],
      y: Double,
      direction: DenseVector[Double]): (DenseVector[Double], DenseVector[Double]) = {

    var lower = x - direction * random.nextDouble() * stepSize
    var upper = lower + direction * stepSize
    val (lowerBound, upperBound) = range

    while ((logp(lower) > y) && (lower.t * direction > lowerBound)) {
      lower -= direction * stepSize
    }

    while ((logp(upper) > y) && upper.t * direction < upperBound) {
      upper += direction * stepSize
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
