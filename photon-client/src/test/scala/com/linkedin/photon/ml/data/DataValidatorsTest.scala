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

import org.testng.Assert
import org.testng.annotations.Test

import com.linkedin.photon.ml.supervised.classification.BinaryClassifier
import com.linkedin.photon.ml.test.CommonTestUtils

class DataValidatorsTest {
  @Test
  def testNonNegativeLabels(): Unit = {
    val vectors = CommonTestUtils.generateDenseFeatureVectors(1, 0, 10)
    val featVec = vectors.head
    Assert.assertTrue(DataValidators.nonNegativeLabels(new LabeledPoint(2.0, featVec)))
    Assert.assertFalse(DataValidators.nonNegativeLabels(new LabeledPoint(-5.0, featVec)))
  }

  @Test
  def testBinaryLabel(): Unit = {
    val vectors = CommonTestUtils.generateDenseFeatureVectors(1, 0, 10)
    val featVec = vectors.head
    Assert.assertTrue(DataValidators.binaryLabel(new LabeledPoint(BinaryClassifier.positiveClassLabel, featVec)))
    Assert.assertTrue(DataValidators.binaryLabel(new LabeledPoint(BinaryClassifier.negativeClassLabel, featVec)))
    Assert.assertFalse(DataValidators.binaryLabel(new LabeledPoint(5.0, featVec)))
    Assert.assertFalse(DataValidators.binaryLabel(new LabeledPoint(-5.0, featVec)))
  }

  @Test
  def testFiniteLabels(): Unit = {
    val vectors = CommonTestUtils.generateDenseFeatureVectors(1, 0, 10)
    val featVec = vectors.head
    Assert.assertTrue(DataValidators.finiteLabel(new LabeledPoint(2.0, featVec)))
    Assert.assertFalse(DataValidators.finiteLabel(new LabeledPoint(Double.NaN, featVec)))
    Assert.assertFalse(DataValidators.finiteLabel(new LabeledPoint(Double.PositiveInfinity, featVec)))
    Assert.assertFalse(DataValidators.finiteLabel(new LabeledPoint(Double.NegativeInfinity, featVec)))
  }

  @Test
  def testFiniteOffset(): Unit = {
    val vectors = CommonTestUtils.generateDenseFeatureVectors(1, 0, 10)
    val featVec = vectors.head
    Assert.assertTrue(DataValidators.finiteOffset(new LabeledPoint(2.0, featVec, 2.5)))
    Assert.assertFalse(DataValidators.finiteOffset(new LabeledPoint(2.0, featVec, Double.NaN)))
    Assert.assertFalse(DataValidators.finiteOffset(new LabeledPoint(2.0, featVec, Double.PositiveInfinity)))
    Assert.assertFalse(DataValidators.finiteOffset(new LabeledPoint(2.0, featVec, Double.NegativeInfinity)))
  }

  @Test
  def testFiniteFeatures(): Unit = {
    val vectors = CommonTestUtils.generateDenseFeatureVectors(1, 1, 10)
    Assert.assertTrue(DataValidators.finiteFeatures(new LabeledPoint(2.0, vectors.head)))
    Assert.assertFalse(DataValidators.finiteFeatures(new LabeledPoint(2.0, vectors.last)))
  }
}