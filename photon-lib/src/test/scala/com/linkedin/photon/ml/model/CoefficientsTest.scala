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
package com.linkedin.photon.ml.model

import breeze.linalg.{DenseVector, SparseVector, Vector}
import org.testng.Assert._
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.constants.MathConst

/**
 * Unit tests for Coefficients.
 */
class CoefficientsTest {

  /**
   *
   * @param values
   * @return
   */
  private def dense(values: Double*) =
    new DenseVector[Double](Array[Double](values: _*))

  /**
   *
   * @param length
   * @param indices
   * @param nnz
   * @return
   */
  private def sparse(length: Int)(indices: Int*)(nnz: Double*) =
    new SparseVector[Double](Array[Int](indices: _*), Array[Double](nnz: _*), length)

  @DataProvider(name = "invalidVectorProvider")
  def makeInvalidVectors(): Array[Array[Vector[Double]]] =
    Array(
      Array(dense(0,0,3,0), sparse(4)(0,2)(0,3)),
      Array(sparse(4)(0,2)(0,3), dense(0,0,3,0)),
      Array(dense(1,2,3), dense(1,2)),
      Array(sparse(2)(1,3)(0,2), sparse(3)(4,5)(0,2))
    )

  @Test(dataProvider = "invalidVectorProvider", expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testPreconditions(v1: Vector[Double], v2: Vector[Double]): Unit =
    new Coefficients(v1, Some(v2))

  @Test
  def testEquals(): Unit = {

    assertFalse(Coefficients(1,0,3,0) == Coefficients(1,0,2,0))
    assertTrue(Coefficients(1,0,3,0) == Coefficients(1,0,3,0))
    assertFalse(Coefficients(4)(0,2)(1,3) == Coefficients(5)(0,2)(1,3))
    assertTrue(Coefficients(4)(0,2)(1,3) == Coefficients(4)(0,2)(1,3))
    assertFalse(Coefficients(1,0,3,0) == Coefficients(4)(0,2)(1,3))
  }

  @Test
  def testComputeScore(): Unit =
    for { v1 <- List(dense(1,0,3,0), sparse(4)(0,2)(1,3))
          v2 <- List(dense(-1,0,0,1), sparse(4)(0,3)(-1,1)) }
      assertEquals(Coefficients(v1).computeScore(v2), v1.dot(v2), MathConst.HIGH_PRECISION_TOLERANCE_THRESHOLD)
}
