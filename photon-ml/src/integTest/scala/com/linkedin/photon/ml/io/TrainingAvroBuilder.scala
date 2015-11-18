/*
 * Copyright 2015 LinkedIn Corp. All rights reserved.
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
package com.linkedin.photon.ml.io

import com.linkedin.photon.avro.generated.FeatureAvro
import java.util.{List => JList}

import org.apache.avro.generic.GenericRecord

/**
 * This defines a common trait for an training avro record builder. Different training input formats should implement
 * the builder differently. This is supposed to be used in tests.
 *
 * @author yizhou
 */
trait TrainingAvroBuilder {
  def setLabel(label: Double): TrainingAvroBuilder

  def setWeight(weight: Double): TrainingAvroBuilder

  def setOffset(offset: Double): TrainingAvroBuilder

  def setFeatures(features: JList[FeatureAvro]): TrainingAvroBuilder

  def build(): GenericRecord
}
