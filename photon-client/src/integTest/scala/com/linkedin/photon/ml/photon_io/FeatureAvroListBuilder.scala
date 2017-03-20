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
package com.linkedin.photon.ml.photon_io

import java.util.{ArrayList => JArrayList, List => JList}

import com.linkedin.photon.avro.generated.FeatureAvro

/**
 * This is a builder that helps build a list of Avro feature items for testing.
 */
class FeatureAvroListBuilder {
  private val features = new JArrayList[FeatureAvro]()

  /**
   *
   * @param name
   * @param term
   * @param value
   * @return
   */
  def append(name: String, term: String, value: Double): FeatureAvroListBuilder = {
    features.add(FeatureAvro.newBuilder()
      .setName(name)
      .setTerm(term)
      .setValue(value)
      .build())
    this
  }

  /**
   *
   * @return
   */
  def build(): JList[FeatureAvro] = features
}
