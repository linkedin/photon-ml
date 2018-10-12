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
package com.linkedin.photon.ml.data

import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.Row
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema
import org.apache.spark.sql.types.DataTypes.DoubleType
import org.apache.spark.sql.types.{StructField, StructType}
import org.testng.Assert.{assertFalse, assertTrue}
import org.testng.annotations.Test

import com.linkedin.photon.ml.constants.MathConst
import com.linkedin.photon.ml.supervised.classification.BinaryClassifier
import com.linkedin.photon.ml.test.CommonTestUtils
import com.linkedin.photon.ml.util.VectorUtils

/**
 * Unit tests for [[DataValidators]].
 */
class DataValidatorsTest {

  import DataValidatorsTest._

  /**
   * Helper function to generate [[org.apache.spark.sql.DataFrame]] rows.
   *
   * @param features Sample features vector
   * @param response Sample response
   * @param offset Sample offset
   * @param weight Sample weight
   * @return Input data grouped into a [[Row]] with the expected schema
   */
  private def getRow(features: Vector, response: Double = 1D, offset: Double = 0D, weight: Double = 1D): Row =
    new GenericRowWithSchema(Array[Any](response, offset, weight, features), ROW_SCHEMA)

  /**
   * Test that [[Row]] column values can be determined as finite or not.
   */
  @Test
  def testRowHasFiniteColumn(): Unit = {

    val featuresVector = VectorUtils.breezeToMl(CommonTestUtils.generateDenseFeatureVectors(1, 0, 10).head)
    val row = getRow(featuresVector, response = 0D, offset = Double.NaN, weight = Double.NaN)

    assertTrue(DataValidators.rowHasFiniteColumn(row, RESPONSE))
    assertFalse(DataValidators.rowHasFiniteColumn(row, OFFSET))
    assertFalse(DataValidators.rowHasFiniteColumn(row, WEIGHT))
    assertTrue(DataValidators.rowHasFiniteColumn(getRow(featuresVector, Double.MaxValue), RESPONSE))
    assertTrue(DataValidators.rowHasFiniteColumn(getRow(featuresVector, Double.MinValue), RESPONSE))
    assertFalse(DataValidators.rowHasFiniteColumn(getRow(featuresVector, Double.PositiveInfinity), RESPONSE))
    assertFalse(DataValidators.rowHasFiniteColumn(getRow(featuresVector, Double.NegativeInfinity), RESPONSE))
  }

  /**
   * Test that [[LabeledPoint]] label values can be determined as finite or not.
   */
  @Test
  def testFiniteLabel(): Unit = {

    val featuresVector = CommonTestUtils.generateDenseFeatureVectors(1, 0, 10).head

    assertTrue(DataValidators.finiteLabel(new LabeledPoint(0D, featuresVector)))
    assertTrue(DataValidators.finiteLabel(new LabeledPoint(Double.MaxValue, featuresVector)))
    assertTrue(DataValidators.finiteLabel(new LabeledPoint(Double.MinValue, featuresVector)))
    assertFalse(DataValidators.finiteLabel(new LabeledPoint(Double.NaN, featuresVector)))
    assertFalse(DataValidators.finiteLabel(new LabeledPoint(Double.PositiveInfinity, featuresVector)))
    assertFalse(DataValidators.finiteLabel(new LabeledPoint(Double.NegativeInfinity, featuresVector)))
  }

  /**
   * Test that [[LabeledPoint]] label values can be determined as binary or not.
   */
  @Test
  def testBinaryLabel(): Unit = {

    val featuresVector = CommonTestUtils.generateDenseFeatureVectors(1, 0, 10).head

    assertTrue(DataValidators.binaryLabel(new LabeledPoint(BinaryClassifier.positiveClassLabel, featuresVector)))
    assertTrue(DataValidators.binaryLabel(new LabeledPoint(BinaryClassifier.negativeClassLabel, featuresVector)))
    assertFalse(DataValidators.binaryLabel(new LabeledPoint(Double.NaN, featuresVector)))
    assertFalse(DataValidators.binaryLabel(new LabeledPoint(0.5, featuresVector)))
    assertFalse(DataValidators.binaryLabel(new LabeledPoint(-1D, featuresVector)))
    assertFalse(DataValidators.binaryLabel(new LabeledPoint(Double.MaxValue, featuresVector)))
    assertFalse(DataValidators.binaryLabel(new LabeledPoint(Double.MinValue, featuresVector)))
    assertFalse(DataValidators.binaryLabel(new LabeledPoint(Double.PositiveInfinity, featuresVector)))
    assertFalse(DataValidators.binaryLabel(new LabeledPoint(Double.NegativeInfinity, featuresVector)))
  }

  /**
   * Test that [[Row]] label column values can be determined as binary or not.
   */
  @Test
  def testRowHasBinaryLabel(): Unit = {

    val featuresVector = VectorUtils.breezeToMl(CommonTestUtils.generateDenseFeatureVectors(1, 0, 10).head)

    assertTrue(DataValidators.rowHasBinaryLabel(getRow(featuresVector), RESPONSE))
    assertTrue(DataValidators.rowHasBinaryLabel(getRow(featuresVector, BinaryClassifier.negativeClassLabel), RESPONSE))
    assertFalse(DataValidators.rowHasBinaryLabel(getRow(featuresVector, Double.NaN), RESPONSE))
    assertFalse(DataValidators.rowHasBinaryLabel(getRow(featuresVector, 0.5), RESPONSE))
    assertFalse(DataValidators.rowHasBinaryLabel(getRow(featuresVector, -1D), RESPONSE))
    assertFalse(DataValidators.rowHasBinaryLabel(getRow(featuresVector, Double.MaxValue), RESPONSE))
    assertFalse(DataValidators.rowHasBinaryLabel(getRow(featuresVector, Double.MinValue), RESPONSE))
    assertFalse(DataValidators.rowHasBinaryLabel(getRow(featuresVector, Double.PositiveInfinity), RESPONSE))
    assertFalse(DataValidators.rowHasBinaryLabel(getRow(featuresVector, Double.NegativeInfinity), RESPONSE))
  }

  /**
   * Test that [[LabeledPoint]] label values can be determined as non-negative or not.
   */
  @Test
  def testNonNegativeLabel(): Unit = {

    val featuresVector = CommonTestUtils.generateDenseFeatureVectors(1, 0, 10).head

    assertTrue(DataValidators.nonNegativeLabel(new LabeledPoint(0D, featuresVector)))
    assertTrue(DataValidators.nonNegativeLabel(new LabeledPoint(Double.MaxValue, featuresVector)))
    assertFalse(DataValidators.nonNegativeLabel(new LabeledPoint(Double.NaN, featuresVector)))
    assertFalse(DataValidators.nonNegativeLabel(new LabeledPoint(Double.MinValue, featuresVector)))
    assertFalse(DataValidators.nonNegativeLabel(new LabeledPoint(Double.PositiveInfinity, featuresVector)))
    assertFalse(DataValidators.nonNegativeLabel(new LabeledPoint(Double.NegativeInfinity, featuresVector)))
  }

  /**
   * Test that [[Row]] label column values can be determined as non-negative or not.
   */
  @Test
  def testRowHasNonNegativeLabel(): Unit = {

    val featuresVector = VectorUtils.breezeToMl(CommonTestUtils.generateDenseFeatureVectors(1, 0, 10).head)

    assertTrue(DataValidators.rowHasNonNegativeLabel(getRow(featuresVector), RESPONSE))
    assertTrue(DataValidators.rowHasNonNegativeLabel(getRow(featuresVector, Double.MaxValue), RESPONSE))
    assertFalse(DataValidators.rowHasNonNegativeLabel(getRow(featuresVector, Double.NaN), RESPONSE))
    assertFalse(DataValidators.rowHasNonNegativeLabel(getRow(featuresVector, Double.MinValue), RESPONSE))
    assertFalse(DataValidators.rowHasNonNegativeLabel(getRow(featuresVector, Double.PositiveInfinity), RESPONSE))
    assertFalse(DataValidators.rowHasNonNegativeLabel(getRow(featuresVector, Double.NegativeInfinity), RESPONSE))
  }

  /**
   * Test that [[LabeledPoint]] feature values can be determined as finite or not.
   */
  @Test
  def testFiniteFeatures(): Unit = {

    val vectors = CommonTestUtils.generateDenseFeatureVectors(1, 1, 10)

    assertTrue(DataValidators.finiteFeatures(new LabeledPoint(2.0, vectors.head)))
    assertFalse(DataValidators.finiteFeatures(new LabeledPoint(2.0, vectors.last)))
  }

  /**
   * Test that [[Row]] feature column values can be determined as finite or not.
   */
  @Test
  def testRowHasFiniteFeatures(): Unit = {

    val vectors = CommonTestUtils.generateDenseFeatureVectors(1, 1, 10)

    assertTrue(DataValidators.rowHasFiniteFeatures(getRow(VectorUtils.breezeToMl(vectors.head)), FEATURES))
    assertFalse(DataValidators.rowHasFiniteFeatures(getRow(VectorUtils.breezeToMl(vectors.last)), FEATURES))
  }

  /**
   * Test that [[LabeledPoint]] offset values can be determined as finite or not.
   */
  @Test
  def testFiniteOffset(): Unit = {

    val featuresVector = CommonTestUtils.generateDenseFeatureVectors(1, 0, 10).head

    assertTrue(DataValidators.finiteOffset(new LabeledPoint(2.0, featuresVector)))
    assertTrue(DataValidators.finiteOffset(new LabeledPoint(2.0, featuresVector, Double.MaxValue)))
    assertTrue(DataValidators.finiteOffset(new LabeledPoint(2.0, featuresVector, Double.MinValue)))
    assertFalse(DataValidators.finiteOffset(new LabeledPoint(2.0, featuresVector, Double.NaN)))
    assertFalse(DataValidators.finiteOffset(new LabeledPoint(2.0, featuresVector, Double.PositiveInfinity)))
    assertFalse(DataValidators.finiteOffset(new LabeledPoint(2.0, featuresVector, Double.NegativeInfinity)))
  }

  /**
   * Test that [[LabeledPoint]] weight values can be determined as valid or not.
   */
  @Test
  def testValidWeight(): Unit = {

    val featuresVector = CommonTestUtils.generateDenseFeatureVectors(1, 0, 10).head

    assertTrue(DataValidators.validWeight(new LabeledPoint(2.0, featuresVector)))
    assertTrue(DataValidators.validWeight(new LabeledPoint(2.0, featuresVector, weight = MathConst.EPSILON * 2)))
    assertTrue(DataValidators.validWeight(new LabeledPoint(2.0, featuresVector, weight = Double.MaxValue)))
    assertFalse(DataValidators.validWeight(new LabeledPoint(2.0, featuresVector, weight = Double.NaN)))
    assertFalse(DataValidators.validWeight(new LabeledPoint(2.0, featuresVector, weight = 0D)))
    assertFalse(DataValidators.validWeight(new LabeledPoint(2.0, featuresVector, weight = MathConst.EPSILON)))
    assertFalse(DataValidators.validWeight(new LabeledPoint(2.0, featuresVector, weight = -1D)))
    assertFalse(DataValidators.validWeight(new LabeledPoint(2.0, featuresVector, weight = Double.MinValue)))
    assertFalse(DataValidators.validWeight(new LabeledPoint(2.0, featuresVector, weight = Double.PositiveInfinity)))
    assertFalse(DataValidators.validWeight(new LabeledPoint(2.0, featuresVector, weight = Double.NegativeInfinity)))
  }

  /**
   * Test that [[Row]] weight column values can be determined as valid or not.
   */
  @Test
  def testRowHasValidWeight(): Unit = {

    val featuresVector = VectorUtils.breezeToMl(CommonTestUtils.generateDenseFeatureVectors(1, 0, 10).head)

    assertTrue(DataValidators.rowHasValidWeight(getRow(featuresVector), WEIGHT))
    assertTrue(DataValidators.rowHasValidWeight(getRow(featuresVector, weight = MathConst.EPSILON * 2), WEIGHT))
    assertTrue(DataValidators.rowHasValidWeight(getRow(featuresVector, weight = Double.MaxValue), WEIGHT))
    assertFalse(DataValidators.rowHasValidWeight(getRow(featuresVector, weight = Double.NaN), WEIGHT))
    assertFalse(DataValidators.rowHasValidWeight(getRow(featuresVector, weight = 0D), WEIGHT))
    assertFalse(DataValidators.rowHasValidWeight(getRow(featuresVector, weight = MathConst.EPSILON), WEIGHT))
    assertFalse(DataValidators.rowHasValidWeight(getRow(featuresVector, weight = -1D), WEIGHT))
    assertFalse(DataValidators.rowHasValidWeight(getRow(featuresVector, weight = Double.MinValue), WEIGHT))
    assertFalse(DataValidators.rowHasValidWeight(getRow(featuresVector, weight = Double.PositiveInfinity), WEIGHT))
    assertFalse(DataValidators.rowHasValidWeight(getRow(featuresVector, weight = Double.NegativeInfinity), WEIGHT))
  }
}

object DataValidatorsTest {

  private val RESPONSE = InputColumnsNames.RESPONSE.toString
  private val OFFSET = InputColumnsNames.OFFSET.toString
  private val WEIGHT = InputColumnsNames.WEIGHT.toString
  private val FEATURES = InputColumnsNames.FEATURES_DEFAULT.toString

  private val RESPONSE_COLUMN = StructField(RESPONSE, DoubleType)
  private val OFFSET_COLUMN = StructField(OFFSET, DoubleType)
  private val WEIGHT_COLUMN = StructField(WEIGHT, DoubleType)
  private val FEATURES_COLUMN = StructField(FEATURES, VectorType)

  private val ROW_SCHEMA = new StructType(Array(RESPONSE_COLUMN, OFFSET_COLUMN, WEIGHT_COLUMN, FEATURES_COLUMN))
}
