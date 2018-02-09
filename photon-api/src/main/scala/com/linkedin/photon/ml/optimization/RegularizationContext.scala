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
package com.linkedin.photon.ml.optimization

import com.linkedin.photon.ml.optimization.RegularizationType.RegularizationType
import com.linkedin.photon.ml.util.{MathUtils, Summarizable}

/**
 * The regularization context holds the information of the regularization type (L1, L2, Elastic net) and the alpha value
 * if the regularization type is elastic net.
 *
 * The regularization term will be lambda [ (1-alpha)/2 ||w||,,2,,^2^ + alpha ||w||,,1,, ]
 *
 * Alpha is computed according to the following rules:
 *
 * <ul>
 *   <li>[[RegularizationType.ELASTIC_NET]] has a default alpha of 0.5</li>
 *   <li>[[RegularizationType.L1]] has a fixed alpha of 1.0</li>
 *   <li>[[RegularizationType.L2]] has a fixed alpha of 0.0</li>
 *   <li>[[RegularizationType.NONE]] has a fixed alpha of 0.0</li>
 * </ul>
 *
 * @param regularizationType The type of regularization
 * @param elasticNetParam The alpha
 */
class RegularizationContext(val regularizationType: RegularizationType, val elasticNetParam: Option[Double])
  extends Summarizable with Serializable {

  checkInvariants()

  /**
   * Invariants that hold for every instance of [[RegularizationContext]].
   */
  def checkInvariants(): Unit = {
    require((regularizationType == RegularizationType.ELASTIC_NET) || elasticNetParam.isEmpty,
      "Elastic net parameter can be specified only for elastic net regularization")

    require(
      regularizationType != RegularizationType.ELASTIC_NET || elasticNetParam.exists(p => 0.0 < p && p <= 1.0),
      s"Elastic net regularization is specified, but elastic net alpha " +
          s"(${elasticNetParam.map(_.toString).getOrElse("NONE")}) is not in interval (0,1).")
  }

  val alpha: Double = (regularizationType, elasticNetParam) match {
    case (RegularizationType.ELASTIC_NET, Some(x)) if x > 0.0d && x <= 1.0d => x
    case (RegularizationType.ELASTIC_NET, None) => 0.5d
    case (RegularizationType.L1, _) => 1.0d
    case (RegularizationType.L2, _) => 0.0d
    case (RegularizationType.NONE, _) => 0.0d
    case _ => throw new IllegalArgumentException(
      s"Wrong input: RegularizationContext($regularizationType, $elasticNetParam)")
  }

  /**
   * Build a human-readable summary for the object.
   *
   * @return A summary of the object in string representation
   */
  override def toSummaryString: String =
    s"regularizationType = $regularizationType" + elasticNetParam.foreach(", elasticNetParam = " + _.toString)

  /**
   * Return the weight for the L1 regularization
   *
   * @param lambda The regularization parameter
   * @return The weight for L1 regularization
   */
  def getL1RegularizationWeight(lambda: Double): Double = alpha * lambda

  /**
   * Return the weight for the L2 regularization
   *
   * @param lambda The regularization parameter
   * @return The weight for L2 regularization
   */
  def getL2RegularizationWeight(lambda: Double): Double = (1 - alpha) * lambda

  /**
   * Indicates whether some other object is "equal to" this one. Two [[RegularizationContext]] objects are equal if
   * they have the same regularization type and elastic net parameter.
   *
   * @param obj The object to compare to this one
   * @return Whether the object is equal to this one
   */
  override def equals(obj: scala.Any): Boolean = obj match {
    case regContext: RegularizationContext =>
      val typeEqual = regContext.regularizationType.equals(this.regularizationType)
      val alphaEqual = (regContext.elasticNetParam, this.elasticNetParam) match {
        case (Some(alpha1), Some(alpha2)) => MathUtils.isAlmostZero(alpha1 - alpha2)
        case (None, None) => true
        case _ => false
      }

      typeEqual && alphaEqual

    case _ => false
  }
}

/**
 * A singleton object for no regularization
 */
object NoRegularizationContext extends RegularizationContext(RegularizationType.NONE, None)

/**
 * A singleton object for L1 regularization
 */
object L1RegularizationContext extends RegularizationContext(RegularizationType.L1, None)

/**
 * A singleton object for L2 regularization
 */
object L2RegularizationContext extends RegularizationContext(RegularizationType.L2, None)

/**
 * A factory object for constructing Elastic Net regularization contexts
 */
object ElasticNetRegularizationContext {

  def apply(alpha: Double): RegularizationContext =
    new RegularizationContext(RegularizationType.ELASTIC_NET, Some(alpha))
}

object RegularizationContext {

  /**
   * Helper method to create a new [[RegularizationContext]] instance.
   *
   * @param regularizationType The type of regularization
   * @param elasticNetParam The alpha
   * @return An instance of RegularizationContext
   */
  def apply(regularizationType: RegularizationType, elasticNetParam: Option[Double] = None): RegularizationContext =
    new RegularizationContext(regularizationType, elasticNetParam)
}
