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
package com.linkedin.photon.ml

import com.linkedin.photon.ml.algorithm.{FixedEffectCoordinate, RandomEffectCoordinateInProjectedSpace}
import com.linkedin.photon.ml.data._
import com.linkedin.photon.ml.function.TwiceDiffFunction
import com.linkedin.photon.ml.model.{FixedEffectModel, RandomEffectModelInProjectedSpace}
import com.linkedin.photon.ml.optimization.LogisticRegressionOptimizationProblem
import com.linkedin.photon.ml.optimization.game.{GLMOptimizationConfiguration, RandomEffectOptimizationProblem}
import com.linkedin.photon.ml.projector.IndexMapProjection
import com.linkedin.photon.ml.supervised.classification.LogisticRegressionModel
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import com.linkedin.photon.ml.test.SparkTestUtils

import org.apache.spark.HashPartitioner
import org.apache.spark.SparkContext

import java.util.concurrent.atomic.AtomicLong

/**
  * A set of utility functions for GAME unit and integration tests
  */
trait GameTestUtils {
  self: SparkTestUtils =>

  /**
    * Default random seed
    */
  var DefaultSeed = 7

  /**
    * Holds the next value in a global id sequence
    */
  var nextId: AtomicLong = new AtomicLong(0)

  /**
    * Adds a global id to the item
    *
    * @param item the item
    * @return a tuple with global id and the item
    */
  def addGlobalId[T](item: T): (Long, T) = (nextId.incrementAndGet, item)

  /**
    * Generates Photon ML labeled points
    *
    * @param size the size of the set to of labeled points to generate
    * @param dimensions the number of dimensions
    * @param seed random seed
    * @return a set of newly generated labeled points
    */
  def generateLabeledPoints(
      size: Int,
      dimensions: Int,
      seed: Int = DefaultSeed) = {

    val data = drawBalancedSampleFromNumericallyBenignDenseFeaturesForBinaryClassifierLocal(
      seed, size, dimensions)

    data.map { case (label, features) =>
      new LabeledPoint(label, features)
    }
  }

  /**
    * Generates a fixed effect dataset
    *
    * @param featureShardId the feature shard id of the dataset
    * @param size the size of the dataset
    * @param dimensions the number of dimensions
    * @param seed random seed
    * @return the newly generated fixed effect dataset
    */
  def generateFixedEffectDataSet(
      featureShardId: String,
      size: Int,
      dimensions: Int,
      seed: Int = DefaultSeed) = {

    val data = sc.parallelize(
      generateLabeledPoints(size, dimensions, seed)
        .map(addGlobalId)
        .toSeq)

    new FixedEffectDataSet(data, featureShardId)
  }

  /**
    * Generates a fixed effect optimization problem
    *
    * @param treeAggregateDepth the Spark treeAggregateDepth setting
    * @param isTrackingState true if state tracking should be enabled
    * @return the newly generated fixed effect optimization problem
    */
  def generateFixedEffectOptimizationProblem(treeAggregateDepth: Int = 1, isTrackingState: Boolean = true) =
    LogisticRegressionOptimizationProblem.buildOptimizationProblem(
      GLMOptimizationConfiguration(), treeAggregateDepth, isTrackingState)

  /**
    * Generates a fixed effect model
    *
    * @param featureShardId the feature shard id of the model
    * @param dimensions the number of dimensions
    * @return the newly generated fixed effect model
    */
  def generateFixedEffectModel(featureShardId: String, dimensions: Int) = new FixedEffectModel(
    sc.broadcast(LogisticRegressionOptimizationProblem.initializeZeroModel(dimensions)),
    featureShardId)

  /**
    * Generates a fixed effect coordinate and model
    *
    * @param featureShardId the feature shard id of the model
    * @param size the size of the dataset
    * @param dimensions the number of dimensions
    * @param seed the random seed
    * @return problem coordinate and random effect model
    */
  def generateFixedEffectCoordinateAndModel(
      featureShardId: String,
      size: Int,
      dimensions: Int,
      seed: Int = DefaultSeed) = {

    val dataset = generateFixedEffectDataSet(featureShardId, size, dimensions, seed)
    val optimizationProblem = generateFixedEffectOptimizationProblem()
    val coordinate = new FixedEffectCoordinate[LogisticRegressionModel, TwiceDiffFunction[LabeledPoint]](
      dataset, optimizationProblem)

    val model = generateFixedEffectModel(featureShardId, dimensions)

    (coordinate, model)
  }

  /**
    * Generates a random effect dataset
    *
    * @param individualIds a set of random effect individual ids
    * @param randomEffectId the random effect id
    * @param featureShardId the feature shard id
    * @param size the size of the dataset
    * @param dimensions the number of dimensions
    * @param seed the random seed
    * @param numPartitions the number of spark partitions
    * @return the newly generated random effect dataset
    */
  def generateRandomEffectDataSet(
      individualIds: Seq[String],
      randomEffectId: String,
      featureShardId: String,
      size: Int,
      dimensions: Int,
      seed: Int = DefaultSeed,
      numPartitions: Int = 4) = {

    val datasets = individualIds.map((_,
      new LocalDataSet(
        generateLabeledPoints(size, dimensions, seed)
          .map(addGlobalId)
          .toArray)))

    val partitioner = new HashPartitioner(numPartitions)
    val globalIdToIndividualIds = sc.parallelize(
      individualIds.map(addGlobalId)).partitionBy(partitioner)
    val activeData = sc.parallelize(datasets).partitionBy(partitioner)

    new RandomEffectDataSet(
      activeData, globalIdToIndividualIds, None, None, randomEffectId, featureShardId)
  }

  /**
    * Generates linear models for random effect models
    *
    * @param individualIds a set of random effect individual ids for which to generate models
    * @param dimensions the number of dimensions
    * @return the newly generated random effect model
    */
  def generateLinearModelsForRandomEffects(
      individualIds: Seq[String],
      dimensions: Int): Seq[(String, GeneralizedLinearModel)] =
    individualIds.map((_, LogisticRegressionOptimizationProblem.initializeZeroModel(dimensions)))

  /**
    * Generates a random effect optimization problem
    *
    * @param dataset the dataset
    * @return the newly generated random effect optimization problem
    */
  def generateRandomEffectOptimizationProblem(dataset: RandomEffectDataSet) = {
    val optimizationProblemBuilder = LogisticRegressionOptimizationProblem.buildOptimizationProblem _

    RandomEffectOptimizationProblem.buildRandomEffectOptimizationProblem[
        LogisticRegressionModel, TwiceDiffFunction[LabeledPoint]](
      optimizationProblemBuilder, GLMOptimizationConfiguration(), dataset)
  }

  /**
    * Generate a random effect coordinate and model
    *
    * @param randomEffectId the random effect id
    * @param featureShardId the feature shard id
    * @param numEntities the number of random effect entities
    * @param size the size of each random effect dataset
    * @param dimensions the number of dimensions of each random effect dataset
    * @param seed the random seed
    * @return problem coordinate and random effect model
    */
  def generateRandomEffectCoordinateAndModel(
      randomEffectId: String,
      featureShardId: String,
      numEntities: Int,
      size: Int,
      dimensions: Int,
      seed: Int = DefaultSeed) = {

    val individualIds = (1 to numEntities).map("re" + _).toSeq

    val randomEffectDataset = generateRandomEffectDataSet(
      individualIds, randomEffectId, featureShardId, size, dimensions, seed)
    val dataset = RandomEffectDataSetInProjectedSpace.buildWithProjectorType(randomEffectDataset, IndexMapProjection)

    val optimizationProblem = generateRandomEffectOptimizationProblem(dataset)
    val coordinate = new RandomEffectCoordinateInProjectedSpace[
        LogisticRegressionModel, TwiceDiffFunction[LabeledPoint]](
      dataset, optimizationProblem)

    val models = sc.parallelize(generateLinearModelsForRandomEffects(individualIds, dimensions))
    val model = new RandomEffectModelInProjectedSpace(
      models, dataset.randomEffectProjector, randomEffectId, featureShardId)

    (coordinate, model)
  }

}
