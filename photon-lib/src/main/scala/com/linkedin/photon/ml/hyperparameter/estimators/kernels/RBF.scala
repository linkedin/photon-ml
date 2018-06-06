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
package com.linkedin.photon.ml.hyperparameter.estimators.kernels

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.exp
import breeze.stats.stddev

/**
 * Implements the radial basis function (RBF) kernel.
 *
 *     $K(x,x') = \exp(-\frac{1}{2} r(x,x')^2)$
 *
 * Where $r(x,x')$ is the Euclidean distance between $x$ and $x'$.
 *
 * @param amplitude the covariance amplitude
 * @param noise the observation noise
 * @param lengthScale the length scale of the kernel. This controls the complexity of the kernel, or the degree to which
 *   it can vary within a given region of the function's domain. Higher values allow less variation, and lower values
 *   allow more.
 */
class RBF(
    amplitude: Double = 1.0,
    noise: Double = 1e-4,
    lengthScale: DenseVector[Double] = DenseVector(1.0))
  extends StationaryKernel(amplitude, noise, lengthScale) {

  /**
   * Computes the RBF kernel function from the pairwise distances between points.
   *
   * @param dists the m x p matrix of pairwise distances between m and p points
   * @return the m x p covariance matrix
   */
  protected[kernels] override def fromPairwiseDistances(dists: DenseMatrix[Double]): DenseMatrix[Double] =
    exp(dists * -0.5)

  /**
   * Creates a new kernel function of the same type, with the given parameters
   *
   * @param theta the parameter vector for the new kernel function
   * @return the new kernel function
   */
  override def withParams(theta: DenseVector[Double]): Kernel = new RBF(
    amplitude = theta(0),
    noise = theta(1),
    lengthScale = theta.slice(2, theta.length))

  /**
   * Builds a kernel with initial settings, based on the observations
   *
   * @param x the observed features
   * @param y the observed labels
   * @return the initial kernel
   */
  override def getInitialKernel(x: DenseMatrix[Double], y: DenseVector[Double]): Kernel =
    new RBF(amplitude = stddev(y))

}
