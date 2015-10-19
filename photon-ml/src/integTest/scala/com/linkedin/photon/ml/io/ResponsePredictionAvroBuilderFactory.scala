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
