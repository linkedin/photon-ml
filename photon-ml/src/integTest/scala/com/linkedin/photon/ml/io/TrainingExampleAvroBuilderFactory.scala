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
