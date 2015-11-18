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

import java.util.{List => JList}

import com.linkedin.photon.avro.generated.{FeatureAvro, TrainingExampleAvro}
import org.apache.avro.generic.GenericRecord

/**
 * This class is a factory that provides builders for building TrainingExampleAvroRecord.
 *
 * It's intended to be used for test purposes.
 *
 * @author yizhou
 */
class TrainingExampleAvroBuilderFactory extends TrainingAvroBuilderFactory {
  import TrainingExampleAvroBuilderFactory._

  def newBuilder(): TrainingAvroBuilder = {
    new TrainingExampleAvroBuilder()
  }
}

private object TrainingExampleAvroBuilderFactory {
  private class TrainingExampleAvroBuilder extends TrainingAvroBuilder {
    private[this] val builder = TrainingExampleAvro.newBuilder()

    override def setLabel(label: Double): TrainingAvroBuilder = {
      builder.setLabel(label)
      this
    }

    override def setFeatures(features: JList[FeatureAvro]): TrainingAvroBuilder = {
      builder.setFeatures(features)
      this
    }

    override def setWeight(weight: Double): TrainingAvroBuilder = {
      builder.setWeight(weight)
      this
    }

    override def build(): GenericRecord = {
      builder.build()
    }

    override def setOffset(offset: Double): TrainingAvroBuilder = {
      builder.setOffset(offset)
      this
    }
  }
}
