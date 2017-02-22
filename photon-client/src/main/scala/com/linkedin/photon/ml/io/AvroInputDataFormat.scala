/*
 * Copyright 2016 LinkedIn Corp. All rights reserved.
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

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.util.IndexMapLoader

/**
 * A general input format supporting reading Avro data.
 */
class AvroInputDataFormat(val suite: GLMSuite) extends InputDataFormat {

  /**
   *
   * @param sc the spark context
   * @param inputPath input path of labeled points
   * @param selectedFeaturesPath optional path of selected features
   * @param minPartitions minimum number of partitions
   * @return an RDD of LabeledPoints
   */
  override def loadLabeledPoints(
        sc: SparkContext,
        inputPath: String,
        selectedFeaturesPath: Option[String],
        minPartitions: Int): RDD[LabeledPoint] = {
    suite.readLabeledPointsFromAvro(sc, inputPath, selectedFeaturesPath, minPartitions)
  }

  /**
   *
   * @return IndexMapLoader
   */
  override def indexMapLoader(): IndexMapLoader = suite.indexMapLoader()

  /**
   *
   * @return
   */
  override def constraintFeatureMap(): Option[Map[Int, (Double, Double)]] = suite.constraintFeatureMap
}
