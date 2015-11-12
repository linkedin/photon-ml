/*
 * Copyright 2014 LinkedIn Corp. All rights reserved.
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

import java.util
import java.io.File

import com.linkedin.photon.avro.generated.FeatureAvro
import org.apache.avro.Schema
import org.apache.avro.generic.{GenericRecordBuilder, GenericData, GenericRecord}

/**
 * This class is a factory that provides builders for building ResponsePredictionAvroRecord.
 *
 * It's intended to be used for test purposes.
 *
 * @author yizhou
 */
class ResponsePredictionAvroBuilderFactory extends TrainingAvroBuilderFactory {
  import ResponsePredictionAvroBuilderFactory._

  override def newBuilder(): TrainingAvroBuilder = new ResponsePredictionAvroBuilder()
}

object ResponsePredictionAvroBuilderFactory {
  val SCHEMA = new Schema.Parser().parse(new File("src/integTest/resources/GLMSuiteIntegTest/ResponsePrediction.avsc"))

  private class ResponsePredictionAvroBuilder extends TrainingAvroBuilder {
    val builder = new GenericRecordBuilder(SCHEMA);

    override def setLabel(label: Double): TrainingAvroBuilder = {
      builder.set(ResponsePredictionFieldNames.response, label)
      this
    }

    override def setFeatures(features: util.List[FeatureAvro]): TrainingAvroBuilder = {
      builder.set(ResponsePredictionFieldNames.features, features)
      this
    }

    override def setWeight(weight: Double): TrainingAvroBuilder = {
      builder.set(ResponsePredictionFieldNames.weight, weight)
      this
    }

    override def build(): GenericRecord = {
      // Add a random extra field to make sure photon does not fail in this situation
      builder.set("randomTestField", "This field is not useful to photon.")
      builder.build()
    }

    override def setOffset(offset: Double): TrainingAvroBuilder = {
      builder.set(ResponsePredictionFieldNames.offset, offset)
      this
    }
  }
}
