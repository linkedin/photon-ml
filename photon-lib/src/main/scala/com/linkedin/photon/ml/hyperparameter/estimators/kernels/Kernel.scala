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

/**
 * Base trait for covariance kernel functions
 *
 * In Gaussian processes estimators and models, the covariance kernel determines the similarity between points in the
 * space. We assume that similarity in domain entails similarity in range, hence the kernel also encodes our prior
 * assumptions about how the function behaves.
 *
 * @see "Gaussian Processes for Machine Learning" (GPML), http://www.gaussianprocess.org/gpml/, Chapter 4
 */
trait Kernel {

  /**
   * Applies the kernel function to the given points
   *
   * @param x the matrix of points, where each of the m rows is a point in the space
   * @return the m x m covariance matrix
   */
  def apply(x: DenseMatrix[Double]): DenseMatrix[Double]

  /**
   * Applies the kernel functions to the two sets of points
   *
   * @param x1 the matrix containing the first set of points, where each of the m rows is a point in the space
   * @param x2 the matrix containing the second set of points, where each of the p rows is a point in the space
   * @return the m x p covariance matrix
   */
  def apply(x1: DenseMatrix[Double], x2: DenseMatrix[Double]): DenseMatrix[Double]

  /**
   * Creates a new kernel function of the same type, with the given parameters
   *
   * @param theta the parameter vector for the new kernel function
   * @return the new kernel function
   */
  def withParams(theta: DenseVector[Double]): Kernel

  /**
   * Returns the kernel parameters as a vector
   *
   * @return the kernel parameters
   */
  def getParams: DenseVector[Double]

  /**
   * Builds a kernel with initial settings, based on the observations
   *
   * @param x the observed features
   * @param y the observed labels
   * @return the initial kernel
   */
  def getInitialKernel(x: DenseMatrix[Double], y: DenseVector[Double]): Kernel

  /**
   * Computes the log likelihood of the kernel parameters
   *
   * @param x the observed features
   * @param y the observed labels
   * @return the log likelihood
   */
  def logLikelihood(x: DenseMatrix[Double], y: DenseVector[Double]): Double

  /**
   * If only one parameter value has been specified, builds a new vector with the single value repeated to fill all
   * dimensions
   *
   * @param param the initial parameters
   * @param dim the dimensions of the final vector
   * @return the vector with all dimensions specified
   */
  def expandDimensions(param: DenseVector[Double], dim: Int): DenseVector[Double] = {
    require(param.length == 1 || param.length == dim,
      "Parameter must contain one global scale or a scale for each feature")

    if (param.length != dim) {
      DenseVector(Array.fill(dim)(param(0)))
    } else {
      param
    }
  }
}
