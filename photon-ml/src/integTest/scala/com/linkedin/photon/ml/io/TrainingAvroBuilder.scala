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
