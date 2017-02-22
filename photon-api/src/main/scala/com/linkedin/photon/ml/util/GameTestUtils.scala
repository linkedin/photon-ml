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
package com.linkedin.photon.ml.util

import java.util.concurrent.atomic.AtomicLong

import org.apache.spark.{HashPartitioner, SparkConf}

import com.linkedin.photon.ml.SparkContextConfiguration
import com.linkedin.photon.ml.algorithm.{FixedEffectCoordinate, RandomEffectCoordinateInProjectedSpace}
import com.linkedin.photon.ml.data._
import com.linkedin.photon.ml.function.glm.{DistributedGLMLossFunction, LogisticLossFunction, SingleNodeGLMLossFunction}
import com.linkedin.photon.ml.model.{Coefficients, FixedEffectModel, RandomEffectModelInProjectedSpace}
import com.linkedin.photon.ml.normalization.NoNormalization
import com.linkedin.photon.ml.optimization.DistributedOptimizationProblem
import com.linkedin.photon.ml.optimization.game.{GLMOptimizationConfiguration, RandomEffectOptimizationProblem}
import com.linkedin.photon.ml.projector.IndexMapProjection
import com.linkedin.photon.ml.supervised.classification.LogisticRegressionModel
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import com.linkedin.photon.ml.test.SparkTestUtils

// TODO: No good way to import test sources between modules - sticking these test utils in main code.

/**
 * A set of utility functions for GAME unit and integration tests.
 */
trait GameTestUtils {

  self: SparkTestUtils =>

  /**
   * Default random seed
   */
  var DefaultSeed = 7

  /**
   * Holds the next value in a unique id sequence
   */
  var nextId: AtomicLong = new AtomicLong(0)

  /**
   * Adds a unique id to the item.
   *
   * @param item The item
   * @return A tuple with global id and the item
   */
  def addUniqueId[T](item: T): (Long, T) = (nextId.incrementAndGet, item)

  /**
   * Generates Photon ML labeled points.
   *
   * @param size The size of the set to of labeled points to generate
   * @param dimensions The number of dimensions
   * @param seed Random seed
   * @return A set of newly generated labeled points
   */
  def generateLabeledPoints(
    size: Int,
    dimensions: Int,
    seed: Int = DefaultSeed): Iterator[LabeledPoint] = {

    val data = drawBalancedSampleFromNumericallyBenignDenseFeaturesForBinaryClassifierLocal(
      seed, size, dimensions)

    data.map { case (label, features) =>
      new LabeledPoint(label, features)
    }
  }

  /**
   * Generates a fixed effect dataset.
   *
   * @param featureShardId The feature shard id of the dataset
   * @param size The number of training samples in the dataset
   * @param dimensions The feature dimension
   * @param seed A random seed
   * @return A newly generated fixed effect dataset
   */
  def generateFixedEffectDataSet(
    featureShardId: String,
    size: Int,
    dimensions: Int,
    seed: Int = DefaultSeed): FixedEffectDataSet = {

    val data = sc.parallelize(generateLabeledPoints(size, dimensions, seed).map(addUniqueId).toSeq)

    new FixedEffectDataSet(data, featureShardId)
  }

  /**
   * Generates a fixed effect optimization problem.
   *
   * @return A newly generated fixed effect optimization problem
   */
  def generateFixedEffectOptimizationProblem: DistributedOptimizationProblem[DistributedGLMLossFunction] = {
    val configuration = GLMOptimizationConfiguration()

    DistributedOptimizationProblem.create(
      configuration,
      DistributedGLMLossFunction.create(
        configuration,
        LogisticLossFunction,
        sc,
        1),
      None,
      LogisticRegressionModel.apply,
      sc.broadcast(NoNormalization()),
      isTrackingState = false,
      isComputingVariance = false)
  }

  /**
   * Generates a fixed effect model.
   *
   * @param featureShardId The feature shard id of the model
   * @param dimensions The model dimension
   * @return The newly generated fixed effect model
   */
  def generateFixedEffectModel(featureShardId: String, dimensions: Int): FixedEffectModel =
    new FixedEffectModel(sc.broadcast(LogisticRegressionModel(Coefficients.initializeZeroCoefficients(dimensions))),
      featureShardId)

  /**
   * Generates a fixed effect coordinate and model.
   *
   * @param featureShardId The feature shard id of the model
   * @param size The number of training samples in the dataset
   * @param dimensions The feature/model dimension
   * @param seed A random seed
   * @return A fixed effect problem coordinate and model
   */
  def generateFixedEffectCoordinateAndModel(
      featureShardId: String,
      size: Int,
      dimensions: Int,
      seed: Int = DefaultSeed)
    : (FixedEffectCoordinate[DistributedGLMLossFunction], FixedEffectModel) = {

    val dataset = generateFixedEffectDataSet(featureShardId, size, dimensions, seed)
    val optimizationProblem = generateFixedEffectOptimizationProblem
    val coordinate = new FixedEffectCoordinate(dataset, optimizationProblem)

    val model = generateFixedEffectModel(featureShardId, dimensions)

    (coordinate, model)
  }

  /**
   * Generates a random effect dataset.
   *
   * @param randomEffectIds A set of random effect IDs
   * @param randomEffectType The random effect type
   * @param featureShardId The feature shard ID
   * @param size The number of training samples in the dataset
   * @param dimensions The feature dimension
   * @param seed The random seed
   * @param numPartitions The number of Spark partitions
   * @return A newly generated random effect dataset
   */
  def generateRandomEffectDataSet(
      randomEffectIds: Seq[String],
      randomEffectType: String,
      featureShardId: String,
      size: Int,
      dimensions: Int,
      seed: Int = DefaultSeed,
      numPartitions: Int = 4): RandomEffectDataSet = {

    val datasets = randomEffectIds.map((_,
      new LocalDataSet(
        generateLabeledPoints(size, dimensions, seed)
          .map(addUniqueId)
          .toArray)))

    val partitioner = new HashPartitioner(numPartitions)
    val uniqueIdToRandomEffectIds = sc.parallelize(
      randomEffectIds.map(addUniqueId)).partitionBy(partitioner)
    val activeData = sc.parallelize(datasets).partitionBy(partitioner)

    new RandomEffectDataSet(
      activeData, uniqueIdToRandomEffectIds, None, None, randomEffectType, featureShardId)
  }

  /**
   * Generates linear models for random effect models.
   *
   * @param randomEffectIds A set of random effect IDs for which to generate models
   * @param dimensions The model dimension
   * @return A newly generated random effect model
   */
  def generateLinearModelsForRandomEffects(
      randomEffectIds: Seq[String],
      dimensions: Int): Seq[(String, GeneralizedLinearModel)] =
    randomEffectIds.map((_, LogisticRegressionModel(Coefficients.initializeZeroCoefficients(dimensions))))

  /**
   * Generates a random effect optimization problem.
   *
   * @param dataset The random effect dataset
   * @return A newly generated random effect optimization problem
   */
  def generateRandomEffectOptimizationProblem(
    dataset: RandomEffectDataSet): RandomEffectOptimizationProblem[SingleNodeGLMLossFunction] = {

    val configuration = GLMOptimizationConfiguration()

    RandomEffectOptimizationProblem.create(
      dataset,
      configuration,
      SingleNodeGLMLossFunction.create(
        configuration,
        LogisticLossFunction),
      LogisticRegressionModel.apply,
      sc.broadcast(NoNormalization()),
      isTrackingState = false,
      isComputingVariance = false)
  }

  /**
   * Generate a random effect coordinate and model.
   *
   * @param randomEffectType The random effect type
   * @param featureShardId The feature shard ID
   * @param numEntities The number of random effect entities
   * @param size The number of training samples per dataset
   * @param dimensions The feature dimension of each dataset
   * @param seed A random seed
   * @return A random effect problem coordinate and model
   */
  def generateRandomEffectCoordinateAndModel(
      randomEffectType: String,
      featureShardId: String,
      numEntities: Int,
      size: Int,
      dimensions: Int,
      seed: Int = DefaultSeed):
        (RandomEffectCoordinateInProjectedSpace[SingleNodeGLMLossFunction], RandomEffectModelInProjectedSpace) = {

    val randomEffectIds = (1 to numEntities).map("re" + _)

    val randomEffectDataset = generateRandomEffectDataSet(
      randomEffectIds,
      randomEffectType,
      featureShardId,
      size,
      dimensions,
      seed)
    val dataset = RandomEffectDataSetInProjectedSpace.buildWithProjectorType(randomEffectDataset, IndexMapProjection)

    val optimizationProblem = generateRandomEffectOptimizationProblem(dataset)
    val coordinate = new RandomEffectCoordinateInProjectedSpace[SingleNodeGLMLossFunction](dataset, optimizationProblem)
    val models = sc.parallelize(generateLinearModelsForRandomEffects(randomEffectIds, dimensions))
    val model = new RandomEffectModelInProjectedSpace(
      models,
      dataset.randomEffectProjector,
      randomEffectType,
      featureShardId)

    (coordinate, model)
  }

  /**
   * Overridden spark test provider that allows for specifying whether to use kryo serialization.
   *
   * @param name the test job name
   * @param body the execution closure
   */
  def sparkTest(name: String, useKryo: Boolean)(body: => Unit) {
    SparkTestUtils.SPARK_LOCAL_CONFIG.synchronized {
      sc = SparkContextConfiguration.asYarnClient(
        new SparkConf().setMaster(SparkTestUtils.SPARK_LOCAL_CONFIG), name, useKryo)

      try {
        body
      } finally {
        sc.stop()
        System.clearProperty("spark.driver.port")
        System.clearProperty("spark.hostPort")
      }
    }
  }
}
