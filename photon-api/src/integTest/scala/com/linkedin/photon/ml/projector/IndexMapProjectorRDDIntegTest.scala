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
package com.linkedin.photon.ml.projector

import breeze.linalg.{DenseVector, Vector}
import org.apache.spark.HashPartitioner
import org.testng.Assert._
import org.testng.annotations.Test

import com.linkedin.photon.ml.data._
import com.linkedin.photon.ml.normalization.{NormalizationContext, NormalizationType}
import com.linkedin.photon.ml.stat.FeatureDataStatistics
import com.linkedin.photon.ml.test.SparkTestUtils
import com.linkedin.photon.ml.util.GameTestUtils

/**
 * Integration tests for [[IndexMapProjectorRDD]].
 */
class IndexMapProjectorRDDIntegTest extends SparkTestUtils with GameTestUtils {

  @Test
  def testBuildIndexMapProjector(): Unit = sparkTest("testBuildIndexMapProjector") {
    val dataSet1 = generateRandomEffectDataSetWithFeatures(
      randomEffectIds = Seq("1"),
      randomEffectType = "per-item",
      featureShardId = "itemShard",
      features = List(
        DenseVector(0.0, 2.0, 3.0, 4.0, 0.0),
        DenseVector(1.0, 5.0, 6.0, 7.0, 0.0)))

    val dataSet2 = generateRandomEffectDataSet(
      randomEffectIds = Seq("1"),
      randomEffectType = "per-item",
      featureShardId = "itemShard",
      size = 20,
      dimensions = 10)

    val projector = IndexMapProjectorRDD.buildIndexMapProjector(dataSet1)
    val projected = projector.projectRandomEffectDataSet(dataSet2)

    val projectedDimentions = projected
      .activeData
      .map { case (_, localDataSet) => localDataSet.dataPoints.head }
      .map { case (_, labeledPoint) => labeledPoint.features.length }
      .take(1)(0)

    assertEquals(projectedDimentions, 4)
  }

  /**
   * Generates a random effect dataset, allowing specific feature vectors.
   *
   * @param randomEffectIds A set of random effect IDs
   * @param randomEffectType The random effect type
   * @param featureShardId The feature shard ID
   * @param features the feature vectors
   * @param numPartitions The number of Spark partitions
   * @return A newly generated random effect dataset
   */
  private def generateRandomEffectDataSetWithFeatures(
      randomEffectIds: Seq[String],
      randomEffectType: String,
      featureShardId: String,
      features: Seq[Vector[Double]],
      numPartitions: Int = 4): RandomEffectDataset = {

    val datasets = randomEffectIds.map((_,
      new LocalDataset(
        features
          .map(vector => new LabeledPoint(0, vector))
          .map(addUniqueId)
          .toArray)))

    val partitioner = new HashPartitioner(numPartitions)
    val uniqueIdToRandomEffectIds = sc.parallelize(
      randomEffectIds.map(addUniqueId)).partitionBy(partitioner)
    val activeData = sc.parallelize(datasets).partitionBy(partitioner)

    new RandomEffectDataset(
      activeData, uniqueIdToRandomEffectIds, None, None, randomEffectType, featureShardId)
  }

  /**
   * Integration tests for [[IndexMapProjectorRDD.projectNormalizationRDD]].
   */
  @Test
  def testProjectionNormalizationContext(): Unit = sparkTest("testNormalizationContextProjector"){

    val features = List(
      DenseVector(0.0, 2.0, 3.0, 4.0, 0.0, 1.0),
      DenseVector(1.0, 5.0, 6.0, 7.0, 0.0, 1.0))

    val projectedSize = features.head.length - 1

    val dataSet = generateRandomEffectDataSetWithFeatures(
      randomEffectIds = Seq("1"),
      randomEffectType = "per-item",
      featureShardId = "itemShard",
      features = features)

    val dataPoints = dataSet.activeData.map{case(_, locals) => locals.dataPoints}
    val localPoints = dataPoints.flatMap{e => e.map(_._2)}

    val summary = FeatureDataStatistics(localPoints, Some(projectedSize))
    val normalizationContext = NormalizationContext(NormalizationType.STANDARDIZATION, summary)

    val projector = IndexMapProjectorRDD.buildIndexMapProjector(dataSet)
    val projectedNormalization = projector.projectNormalizationRDD(normalizationContext)

    val projectedDimensions = projectedNormalization.mapValues(_.size).take(1)(0)._2

    assertEquals(projectedDimensions, projectedSize)
  }

  @Test
  def testProjectCoefficientsRDD(): Unit = sparkTest("testProjectCoefficientsRDD") {
    val features = List(
      DenseVector(0.0, 2.0, 3.0, 4.0, 0.0, 1.0),
      DenseVector(1.0, 5.0, 6.0, 7.0, 0.0, 1.0))

    val originalSize = features.head.length
    val projectedSize = originalSize - 1

    val dataSet = generateRandomEffectDataSetWithFeatures(
      randomEffectIds = Seq("1"),
      randomEffectType = "per-item",
      featureShardId = "itemShard",
      features = features)

    val projector = IndexMapProjectorRDD.buildIndexMapProjector(dataSet)
    val reIds = dataSet.activeData.map(_._1).collect

    // The model contains an re id from a prior run that doesn't exist in current data
    val reIdsWithExtra = reIds :+ "extraReId"
    val models = sc.parallelize(generateLinearModelsForRandomEffects(reIdsWithExtra, projectedSize))

    val projected = projector.projectCoefficientsRDD(models).collect

    // Ensure that re id present in the model but not in the dataset is preserved
    assertEquals(projected.map(_._1).toSet, reIdsWithExtra.toSet)

    projected.dropRight(1).foreach { case (_, model) =>
      assertEquals(model.coefficients.means.length, originalSize)
    }

    // The "extra" model doesn't have a projector, so its size remains the same
    assertEquals(projected.last._2.coefficients.means.length, projectedSize)
  }

  @Test
  def testTransformCoefficientsRDD(): Unit = sparkTest("testTansformCoefficientsRDD") {
    val features = List(
      DenseVector(0.0, 2.0, 3.0, 4.0, 0.0, 1.0),
      DenseVector(1.0, 5.0, 6.0, 7.0, 0.0, 1.0))

    val originalSize = features.head.length
    val projectedSize = originalSize - 1

    val dataSet = generateRandomEffectDataSetWithFeatures(
      randomEffectIds = Seq("1"),
      randomEffectType = "per-item",
      featureShardId = "itemShard",
      features = features)

    val projector = IndexMapProjectorRDD.buildIndexMapProjector(dataSet)
    val reIds = dataSet.activeData.map(_._1).collect

    // The model contains an re id from a prior run that doesn't exist in current data
    val reIdsWithExtra = reIds :+ "extraReId"
    val models = sc.parallelize(generateLinearModelsForRandomEffects(reIdsWithExtra, originalSize))

    val projected = projector.transformCoefficientsRDD(models).collect

    // Ensure that re id present in the model but not in the dataset is preserved
    assertEquals(projected.map(_._1).toSet, reIdsWithExtra.toSet)

    projected.dropRight(1).foreach { case (_, model) =>
      assertEquals(model.coefficients.means.length, projectedSize)
    }

    // The "extra" model doesn't have a projector, so its size remains the same
    assertEquals(projected.last._2.coefficients.means.length, originalSize)
  }
}
