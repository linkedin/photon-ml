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
package com.linkedin.photon.ml.io.deprecated

import java.util.{List => JList}

import org.apache.avro.generic.GenericRecord

import com.linkedin.photon.avro.generated.FeatureAvro

/**
 * This defines a common trait for an training avro record builder. Different training input formats should implement
 * the builder differently. This is supposed to be used in tests.
 */
trait TrainingAvroBuilder {
  /**
   *
   * @param label
   * @return
   */
  def setLabel(label: Double): TrainingAvroBuilder

  /**
   *
   * @param weight
   * @return
   */
  def setWeight(weight: Double): TrainingAvroBuilder

  /**
   *
   * @param offset
   * @return
   */
  def setOffset(offset: Double): TrainingAvroBuilder

  /**
   *
   * @param features
   * @return
   */
  def setFeatures(features: JList[FeatureAvro]): TrainingAvroBuilder

  /**
   *
   * @return
   */
  def build(): GenericRecord
}
