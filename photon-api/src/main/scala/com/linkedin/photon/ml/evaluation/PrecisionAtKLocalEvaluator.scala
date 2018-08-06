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
package com.linkedin.photon.ml.evaluation

import java.lang.{Double => JDouble}
import java.util.{Arrays => JArrays, Comparator => JComparator}

import com.linkedin.photon.ml.constants.MathConst

/**
 * Evaluator for Precision@k. Definition of this evaluator along with terminologies used here can be
 * found at [[https://en.wikipedia.org/wiki/Information_retrieval#Precision_at_K]]. No special tiebreaker is used if
 * there are ties in scores; one of them will be picked randomly.
 *
 * @param k The cut-off rank for which precision is computed
 */
protected[evaluation] class PrecisionAtKLocalEvaluator(protected val k: Int) extends LocalEvaluator {

  require(k > 0, s"Position k must be greater than 0: $k")

  /**
   * Compute the precision @ k for the given data.
   *
   * @param scoresAndLabelsAndWeights An [[Array]] of (score, label, weight) tuples
   * @return The precision @ k
   */
  protected[evaluation] override def evaluate(scoresAndLabelsAndWeights: Array[(Double, Double, Double)]): Double = {

    // Directly calling scoresAndLabelAndWeight.sort in Scala would cause some performance issue with large arrays
    val comparator = new JComparator[(Double, Double, Double)]() {
      override def compare(tuple1: (Double, Double, Double), tuple2: (Double, Double, Double)): Int = {
        JDouble.compare(tuple1._1, tuple2._1)
      }
    }
    JArrays.sort(scoresAndLabelsAndWeights, comparator.reversed())

    val hits = scoresAndLabelsAndWeights.take(k).count(_._2 > MathConst.POSITIVE_RESPONSE_THRESHOLD)

    1.0 * hits / k
  }

  //
  // Object functions
  //

  /**
   * Compares two [[PrecisionAtKLocalEvaluator]] objects.
   *
   * @param other Some other object
   * @return True if the both models conform to the equality contract and have the same model coefficients, false
   *         otherwise
   */
  override def equals(other: Any): Boolean = other match {
    case that: PrecisionAtKLocalEvaluator => this.k == that.k
    case _ => false
  }

  /**
   * Returns a hash code value for the object.
   *
   * @return An [[Int]] hash code
   */
  override def hashCode: Int = k.hashCode()
}
