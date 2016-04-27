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
package com.linkedin.photon.ml.avro

import scala.collection.Map

import breeze.linalg.{SparseVector, DenseVector}
import org.testng.Assert.assertEquals
import org.testng.annotations.Test

import com.linkedin.photon.ml.avro.data.NameAndTerm
import com.linkedin.photon.ml.model.Coefficients


/**
 * Simple tests for functions in [[AvroUtils]]
 */
class AvroUtilsTest {

  @Test
  def testCoefficientsAndBayesianLinearModelAvroRecordConversion(): Unit = {

    // Initialize the coefficients and related meta data
    val length = 4
    val sparseVector = new SparseVector[Double](index = Array(1), data = Array(1.0), length = length)
    val sparseCoefficients = Coefficients(sparseVector)
    val denseVector = new DenseVector[Double](data = Array(0.0, 1.0, 2.0, 3.0))
    val denseCoefficients = Coefficients(denseVector)
    val modelId = "modelId"
    val intToNameAndTermMap = Map(0 -> NameAndTerm("0", "0"), 1 -> NameAndTerm("1", "1"), 2 -> NameAndTerm("2", "2"),
      3 -> NameAndTerm("3", "3"))
    val nameAndTermToIntMap = intToNameAndTermMap.map(_.swap)

    // Convert the sparse coefficients to Avro record, and convert it back to coefficients
    val sparseCoefficientsAvro = AvroUtils.convertCoefficientsToBayesianLinearModelAvroRecord(sparseCoefficients,
      modelId, intToNameAndTermMap)
    val recoveredSparseVector = AvroUtils.parseMeanVectorFromBayesianLinearModelAvroRecord(sparseCoefficientsAvro,
      nameAndTermToIntMap)
    val recoveredSparseCoefficients = Coefficients(recoveredSparseVector)
    assertEquals(sparseCoefficients, recoveredSparseCoefficients)

    // Convert the dense coefficients to Avro record, and convert it back to coefficients
    val denseCoefficientsAvro = AvroUtils.convertCoefficientsToBayesianLinearModelAvroRecord(denseCoefficients,
      modelId, intToNameAndTermMap)
    val recoveredDenseVector = AvroUtils.parseMeanVectorFromBayesianLinearModelAvroRecord(denseCoefficientsAvro,
      nameAndTermToIntMap)
    val recoveredDenseCoefficients = Coefficients(recoveredDenseVector)
    assertEquals(denseCoefficients, recoveredDenseCoefficients)
  }
}
