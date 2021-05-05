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
package com.linkedin.photon.ml.data.avro

import breeze.linalg.{DenseVector, SparseVector}
import org.testng.Assert.assertEquals
import org.testng.annotations.Test

import com.linkedin.photon.ml.index.DefaultIndexMap
import com.linkedin.photon.ml.model.Coefficients
import com.linkedin.photon.ml.supervised.classification.LogisticRegressionModel
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import com.linkedin.photon.ml.util._

/**
 * Simple tests for functions in [[AvroUtils]].
 */
class AvroUtilsTest {

  // Test both the convertGLMModelToBayesianLinearModelAvro and readMeanOfCoefficientsFromBayesianLinearModelAvro
  // functions
  @Test
  def testGLMModelAndBayesianLinearModelAvroRecordConversion(): Unit = {

    // Initialize the coefficients and related meta data
    val length = 4
    val sparseVector = new SparseVector[Double](index = Array(1), data = Array(1.0), length = length)
    val sparseCoefficients = Coefficients(sparseVector)
    val denseVector = new DenseVector[Double](data = Array(0.0, 1.0, 2.0, 3.0))
    val denseCoefficients = Coefficients(denseVector)
    val modelId = "modelId"
    val indexMap = new DefaultIndexMap(
      Map(
        Utils.getFeatureKey("0", "0") -> 0,
        Utils.getFeatureKey("1", "1") -> 1,
        Utils.getFeatureKey("2", "2") -> 2,
        Utils.getFeatureKey("3", "3") -> 3)
        .toMap)

    val sparseGlm: GeneralizedLinearModel = new LogisticRegressionModel(sparseCoefficients)

    // Convert the sparse coefficients to Avro record, and convert it back to coefficients
    val sparseCoefficientsAvro = AvroUtils.convertGLMModelToBayesianLinearModelAvro(sparseGlm,
      modelId, indexMap)
    val recoveredSparseGlm = AvroUtils.convertBayesianLinearModelAvroToGLM(sparseCoefficientsAvro, indexMap)

    val Z: Coefficients = recoveredSparseGlm.coefficients
    val Z1: Coefficients = sparseCoefficients
    assertEquals(Z1, Z)

    val denseGlm: GeneralizedLinearModel = new LogisticRegressionModel(denseCoefficients)

    // Convert the dense coefficients to Avro record, and convert it back to coefficients
    val denseCoefficientsAvro = AvroUtils.convertGLMModelToBayesianLinearModelAvro(denseGlm,
      modelId, indexMap)
    val recoveredDenseGlm = AvroUtils.convertBayesianLinearModelAvroToGLM(denseCoefficientsAvro, indexMap)

    assertEquals(denseCoefficients, recoveredDenseGlm.coefficients)
  }
}
