package com.linkedin.photon.ml.io

import java.util.{List => JList, ArrayList => JArrayList}

import com.linkedin.photon.avro.generated.FeatureAvro

/**
 * This is a builder that helps build a list of Avro feature items for testing.
 *
 * @author yizhou
 */
class FeatureAvroListBuilder {
    private val features = new JArrayList[FeatureAvro]()

    def append(name: String, term: String, value: Double): FeatureAvroListBuilder = {
      features.add(FeatureAvro.newBuilder()
        .setName(name)
        .setTerm(term)
        .setValue(value)
        .build())
      this
    }

    def build(): JList[FeatureAvro] = features
}
