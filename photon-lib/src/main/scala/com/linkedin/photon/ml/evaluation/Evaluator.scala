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

import org.apache.spark.rdd.RDD

/**
 * An interface for evaluation metric computation implementations at the [[RDD]] level.
 */
trait Evaluator {

  type ScoredData

  val evaluatorType: EvaluatorType

  //
  // Evaluator functions
  //

  /**
   * Get the name of this [[Evaluator]] object.
   *
   * @return The name of this [[Evaluator]].
   */
  def getEvaluatorName: String = evaluatorType.name

  /**
   * Compute the evaluation metric for the given data.
   *
   * @param scoresAndLabelsAndWeights A [[RDD]] of scored data
   * @return The evaluation metric
   */
  protected[ml] def evaluate(scoresAndLabelsAndWeights: RDD[ScoredData]): Double

  //
  // Object functions
  //

  /**
   * Compares two [[Evaluator]] objects.
   *
   * @param other Some other object
   * @return True if the both models conform to the equality contract and have the same model coefficients, false
   *         otherwise
   */
  override def equals(other: Any): Boolean = other match {
    case that: Evaluator => this.evaluatorType == that.evaluatorType
    case _ => false
  }

  /**
   * Returns a hash code value for the object.
   *
   * @return An [[Int]] hash code
   */
  override def hashCode: Int = evaluatorType.hashCode()
}
