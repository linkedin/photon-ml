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
package com.linkedin.photon.ml.model

import breeze.linalg.{SparseVector, DenseVector, Vector}
import org.testng.Assert._
import org.testng.annotations.Test

import com.linkedin.photon.ml.constants.MathConst


/**
 * @author xazhang
 */
class CoefficientsTest {

  private def getVector(indices: Array[Int], values: Array[Double], length: Int, isDenseVector: Boolean)
  : Vector[Double] = {

    if (isDenseVector) {
      val data = new Array[Double](length)
      indices.zip(values).foreach { case (index, value) => data(index) = value }
      new DenseVector[Double](data)
    } else {
      val (sortedIndices, sortedValues) = indices.zip(values).sortBy(_._1).unzip
      new SparseVector[Double](sortedIndices.toArray, sortedValues.toArray, length)
    }
  }

  @Test
  def testComputeScore(): Unit = {

    val length = 4

    // Coefficients
    val meansIndices = Array(0, 2)
    val meansValues = Array(1.0, 3.0)
    val meansAsDenseVector = getVector(meansIndices, meansValues, length, isDenseVector = true)
    val denseCoefficients = Coefficients(meansAsDenseVector)
    val meansAsSparseVector = getVector(meansIndices, meansValues, length, isDenseVector = false)
    val sparseCoefficients = Coefficients(meansAsSparseVector)

    // Features
    val featuresIndices = Array(0, 3)
    val featuresValues = Array(-1.0, 1.0)
    val featuresAsDenseVector = getVector(featuresIndices, featuresValues, length, isDenseVector = true)
    val featuresAsSparseVector = getVector(featuresIndices, featuresValues, length, isDenseVector = false)

    val groundTruthScore = -1.0
    assertEquals(denseCoefficients.computeScore(featuresAsDenseVector), groundTruthScore,
      MathConst.HIGH_PRECISION_TOLERANCE_THRESHOLD)
    assertEquals(denseCoefficients.computeScore(featuresAsSparseVector), groundTruthScore,
      MathConst.HIGH_PRECISION_TOLERANCE_THRESHOLD)
    assertEquals(sparseCoefficients.computeScore(featuresAsDenseVector), groundTruthScore,
      MathConst.HIGH_PRECISION_TOLERANCE_THRESHOLD)
    assertEquals(sparseCoefficients.computeScore(featuresAsSparseVector), groundTruthScore,
      MathConst.HIGH_PRECISION_TOLERANCE_THRESHOLD)
  }

  def testEquals(): Unit = {
    val length = 2
    val meansIndices = Array(0, 1)
    val meansValues = Array(1.0, -1.0)

    // Coefficients with dense and sparse vectors should be different
    val meansAsDenseVector = getVector(meansIndices, meansValues, length, isDenseVector = true)
    val denseCoefficients = Coefficients(meansAsDenseVector)
    val meansAsSparseVector = getVector(meansIndices, meansValues, length, isDenseVector = false)
    val sparseCoefficients = Coefficients(meansAsSparseVector)
    assertTrue(!denseCoefficients.equals(sparseCoefficients))

    // Coefficients with different data should be different
    val meansIndices2 = Array(0, 1)
    val meansValues2 = Array(1.0, 1.0)
    val meansAsDenseVector2 = getVector(meansIndices2, meansValues2, length, isDenseVector = true)
    val denseCoefficients2 = Coefficients(meansAsDenseVector2)
    val meansAsSparseVector2 = getVector(meansIndices2, meansValues2, length, isDenseVector = false)
    val sparseCoefficients2 = Coefficients(meansAsSparseVector2)
    assertTrue(!denseCoefficients.equals(denseCoefficients2))
    assertTrue(!sparseCoefficients.equals(sparseCoefficients2))

    // Coefficients with same data and same type should be the same
    val meansIndices3 = meansIndices.clone()
    val meansValues3 = meansValues.clone()
    val meansAsDenseVector3 = getVector(meansIndices3, meansValues3, length, isDenseVector = true)
    val denseCoefficients3 = Coefficients(meansAsDenseVector3)
    val meansAsSparseVector3 = getVector(meansIndices3, meansValues3, length, isDenseVector = false)
    val sparseCoefficients3 = Coefficients(meansAsSparseVector3)
    assertTrue(denseCoefficients.equals(denseCoefficients3))
    assertTrue(sparseCoefficients.equals(sparseCoefficients3))
  }
}
