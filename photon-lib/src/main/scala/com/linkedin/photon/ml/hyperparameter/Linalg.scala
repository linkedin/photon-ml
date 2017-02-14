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
   * @param l the Cholesky factorization of coefficient matrix A
   * @param b the right-hand side vector b
   * @return the solution
   */
  protected[hyperparameter] def choleskySolve(
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
    lapack.dpptrs("L", n, 1, lArr, bArr, n, info)

    if (info.`val` < 0) {
      throw new RuntimeException(
        s"The value at position ${-info.`val`} had an illegal value.")
    }

    DenseVector(bArr)
  }

  /**
   * Computes the mean vector of a collection of vectors
   *
   * @param vectors the input vectors
   * @return the vector mean
   */
  protected[hyperparameter] def vectorMean(vectors: Seq[DenseVector[Double]]): DenseVector[Double] = {
    val totals = vectors.reduce(_ + _)
    totals / vectors.length.toDouble
  }
}
