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
package com.linkedin.photon.ml.util

import breeze.linalg.{DenseMatrix, DenseVector}
import com.github.fommil.netlib.LAPACK.{getInstance => lapack}
import org.netlib.util.intW

/**
 * A set of linear algebra helper functions
 */
object Linalg {

  /**
   * Solves the system of linear equations $Ax = b$, given the Cholesky factorization $l$ of $A$.
   *
   * @param l The Cholesky factorization of coefficient matrix A
   * @param b The right-hand side vector b
   * @return The solution
   */
  protected[ml] def choleskySolve(
      l: DenseMatrix[Double],
      b: DenseVector[Double]): DenseVector[Double] = {

    val n = b.length

    // Convert the right-hand side to an array. Lapack will write the result to this array
    val bArr = b.toArray

    // Pack the lower triangular part of the factorization into the format that Lapack expects
    val lArr = (0 until l.cols).flatMap { j =>
      (j until n).map { i => l(i, j) }
    }.toArray

    val info = new intW(0)

    // The LAPACK "dpptrs" function solves $Ax = b$, given the Cholesky factorization of $A$. The result is written to
    // bArr.
    //
    // See:
    //   http://www.netlib.org/lapack/explore-html/da/dba/group__double_o_t_h_e_rcomputational_gaa0b8f7830a459c434c84ce5e7a939850.html#gaa0b8f7830a459c434c84ce5e7a939850
    lapack.dpptrs("L", n, 1, lArr, bArr, n, info)

    if (info.`val` < 0) {
      throw new RuntimeException(
        s"The value at position ${-info.`val`} had an illegal value.")
    }

    DenseVector(bArr)
  }

  /**
   * Inverts the matrix $A$, given the Cholesky factorization $l$ of $A$.
   *
   * @param l The Cholesky factorization of matrix A
   * @return The solution
   */
  protected[ml] def choleskyInverse(l: DenseMatrix[Double]): DenseMatrix[Double] = {

    val n = l.rows

    // Lapack will write the result to this array
    val lArr = l.toArray

    val info = new intW(0)

    // The LAPACK "dpotri" function inverts a matrix $A$, given the Cholesky factorization $l$ of $A$. The results are
    // written to lArr.
    //
    // See:
    //   http://www.netlib.org/lapack/explore-html/d1/d7a/group__double_p_ocomputational_ga9dfc04beae56a3b1c1f75eebc838c14c.html
    lapack.dpotri("L", n, lArr, n, info)

    if (info.`val` < 0) {
      throw new RuntimeException(
        s"The value at position ${-info.`val`} had an illegal value.")
    }

    DenseMatrix(lArr).reshape(n, n)
  }

  /**
   * Computes the mean vector of a collection of vectors
   *
   * @param vectors The input vectors
   * @return The vector mean
   */
  protected[ml] def vectorMean(vectors: Seq[DenseVector[Double]]): DenseVector[Double] = {
    val totals = vectors.reduce(_ + _)
    totals / vectors.length.toDouble
  }
}
