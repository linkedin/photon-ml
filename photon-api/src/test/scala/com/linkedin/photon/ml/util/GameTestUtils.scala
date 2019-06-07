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
package com.linkedin.photon.ml.util

import java.util.concurrent.atomic.AtomicLong

import org.apache.spark.{HashPartitioner, SparkConf}
import breeze.linalg.DenseVector
import org.testng.annotations.DataProvider

import com.linkedin.photon.ml.{SparkSessionConfiguration, TaskType}
import com.linkedin.photon.ml.Types.{FeatureShardId, REId, REType, UniqueSampleId}
import com.linkedin.photon.ml.algorithm.{FixedEffectCoordinate, RandomEffectCoordinate}
import com.linkedin.photon.ml.data._
import com.linkedin.photon.ml.function.glm.{DistributedGLMLossFunction, LogisticLossFunction, SingleNodeGLMLossFunction}
import com.linkedin.photon.ml.model.{Coefficients, FixedEffectModel, RandomEffectModel}
import com.linkedin.photon.ml.normalization.NoNormalization
import com.linkedin.photon.ml.optimization.OptimizerType.OptimizerType
import com.linkedin.photon.ml.optimization._
import com.linkedin.photon.ml.optimization.game.{FixedEffectOptimizationConfiguration, RandomEffectOptimizationConfiguration, RandomEffectOptimizationProblem}
import com.linkedin.photon.ml.projector.LinearSubspaceProjector
import com.linkedin.photon.ml.supervised.classification.LogisticRegressionModel
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import com.linkedin.photon.ml.test.SparkTestUtils

/**
 * A set of utility functions for GAME unit and integration tests.
 */
trait GameTestUtils extends SparkTestUtils {

  /**
   *
   */
  val defaultTaskType = TaskType.LOGISTIC_REGRESSION

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
   * A trivial set of fixed labeled points for simple tests to verify by hand.
   *
   * @return A single set of 10 vectors of 2 features and a label.
   */
  @DataProvider
  def trivialLabeledPoints(): Array[Array[Seq[LabeledPoint]]] = Array(
    Array(
      Seq(
        LabeledPoint(0.0, DenseVector(-0.7306653538519616, 0.0)),
        LabeledPoint(1.0, DenseVector(0.6750417712898752, -0.4232874171873786)),
        LabeledPoint(1.0, DenseVector(0.1863463229359709, -0.8163423997075965)),
        LabeledPoint(0.0, DenseVector(-0.6719842051493347, 0.0)),
        LabeledPoint(1.0, DenseVector(0.9699938346531928, 0.0)),
        LabeledPoint(1.0, DenseVector(0.22759406190283604, 0.0)),
        LabeledPoint(1.0, DenseVector(0.9688721028330911, 0.0)),
        LabeledPoint(0.0, DenseVector(0.5993795346650845, 0.0)),
        LabeledPoint(0.0, DenseVector(0.9219423508390701, -0.8972778242305388)),
        LabeledPoint(0.0, DenseVector(0.7006904841584055, -0.5607635619919824)))))

  /**
   * Generates an optimizer configuration.
   *
   * @param optimizer The optimizer type
   * @param maxIterations The upper limit on the number of optimization iterations to perform
   * @param tolerance The relative tolerance limit for optimization
   * @return A newly generated [[OptimizerConfig]]
   */
  def generateOptimizerConfig(
      optimizer: OptimizerType = OptimizerType.LBFGS,
      maxIterations: Int = 80,
      tolerance: Double = 1e-6): OptimizerConfig =
    OptimizerConfig(optimizer, maxIterations, tolerance, constraintMap = None)

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
  def generateFixedEffectDataset(
      featureShardId: FeatureShardId,
      size: Int,
      dimensions: Int,
      seed: Int = DefaultSeed): FixedEffectDataset = {

    val data = sc.parallelize(generateLabeledPoints(size, dimensions, seed).map(addUniqueId).toSeq)

    new FixedEffectDataset(data, featureShardId)
  }

  /**
   * Generates a fixed effect optimization problem.
   *
   * @return A newly generated fixed effect optimization problem
   */
  def generateFixedEffectOptimizationProblem: DistributedOptimizationProblem[DistributedGLMLossFunction] = {

    val configuration = FixedEffectOptimizationConfiguration(generateOptimizerConfig())

    DistributedOptimizationProblem(
      configuration,
      DistributedGLMLossFunction(configuration, LogisticLossFunction, treeAggregateDepth = 1),
      None,
      LogisticRegressionModel.apply,
      PhotonBroadcast(sc.broadcast(NoNormalization())),
      VarianceComputationType.NONE,
      isTrackingState = false)
  }

  /**
   * Generates a fixed effect model.
   *
   * @param featureShardId The feature shard id of the model
   * @param dimensions The model dimension
   * @return The newly generated fixed effect model
   */
  def generateFixedEffectModel(featureShardId: FeatureShardId, dimensions: Int): FixedEffectModel =
    new FixedEffectModel(
      sc.broadcast(LogisticRegressionModel(Coefficients.initializeZeroCoefficients(dimensions))),
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
      featureShardId: FeatureShardId,
      size: Int,
      dimensions: Int,
      seed: Int = DefaultSeed): (FixedEffectCoordinate[DistributedGLMLossFunction], FixedEffectModel) = {

    val dataset = generateFixedEffectDataset(featureShardId, size, dimensions, seed)
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
  def generateRandomEffectDataset(
      randomEffectIds: Seq[REId],
      randomEffectType: REType,
      featureShardId: FeatureShardId,
      size: Int,
      dimensions: Int,
      seed: Int = DefaultSeed,
      numPartitions: Int = 4): RandomEffectDataset = {

    val projectedDimensions: Int = dimensions / 2
    val indices = (0 until dimensions by 2).toSet
    val subspaceProjector = new LinearSubspaceProjector(indices, dimensions)

    val datasets = randomEffectIds.map { rEID =>
      (rEID, new LocalDataset(generateLabeledPoints(size, projectedDimensions, seed).map(addUniqueId).toArray))
    }
    val projectors = datasets.map { case (rEID, _) =>
      (rEID, subspaceProjector)
    }

    val partitionMap = randomEffectIds
      .zipWithIndex
      .map { case (rEId, index) =>
        (rEId, index % numPartitions)
      }
      .toMap
    val rEPartitioner = new RandomEffectDatasetPartitioner(numPartitions, sc.broadcast(partitionMap))
    val hashPartitioner = new HashPartitioner(numPartitions)
    val activeData = sc.parallelize(datasets).partitionBy(rEPartitioner)
    val passiveData = sc.emptyRDD[(UniqueSampleId, (REId, LabeledPoint))].partitionBy(hashPartitioner)
    val uniqueIdToRandomEffectIds = sc.parallelize(randomEffectIds.map(addUniqueId)).partitionBy(hashPartitioner)
    val projectorsRDD = sc.parallelize(projectors).partitionBy(hashPartitioner)

    new RandomEffectDataset(
      activeData,
      passiveData,
      uniqueIdToRandomEffectIds,
      projectorsRDD,
      randomEffectType,
      featureShardId)
  }

  /**
   * Generates linear models for random effect models.
   *
   * @param randomEffectIds A set of random effect IDs for which to generate models
   * @param dimensions The model dimension
   * @return A newly generated random effect model
   */
  def generateLinearModelsForRandomEffects(
      randomEffectIds: Seq[REId],
      dimensions: Int): Seq[(REId, GeneralizedLinearModel)] =
    randomEffectIds.map((_, LogisticRegressionModel(Coefficients.initializeZeroCoefficients(dimensions))))

  /**
   * Generates a random effect optimization problem.
   *
   * @param dataset The random effect dataset
   * @return A newly generated random effect optimization problem
   */
  def generateRandomEffectOptimizationProblem(
      dataset: RandomEffectDataset): RandomEffectOptimizationProblem[SingleNodeGLMLossFunction] = {

    val isTrackingState = false
    val configuration = RandomEffectOptimizationConfiguration(generateOptimizerConfig())
    val normalizationBroadcast = sc.broadcast(NoNormalization())
    val randomEffectOptimizationProblems = dataset
      .activeData
      .mapValues { _ =>
        SingleNodeOptimizationProblem(
          configuration,
          SingleNodeGLMLossFunction(configuration, LogisticLossFunction),
          LogisticRegressionModel.apply,
          PhotonBroadcast(normalizationBroadcast),
          VarianceComputationType.NONE,
          isTrackingState)
      }

    new RandomEffectOptimizationProblem(randomEffectOptimizationProblems, LogisticRegressionModel.apply, isTrackingState)
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
      randomEffectType: REType,
      featureShardId: FeatureShardId,
      numEntities: Int,
      size: Int,
      dimensions: Int,
      seed: Int = DefaultSeed): (RandomEffectCoordinate[SingleNodeGLMLossFunction], RandomEffectModel) = {

    val randomEffectIds = (1 to numEntities).map("re" + _)

    val randomEffectDataset = generateRandomEffectDataset(
      randomEffectIds,
      randomEffectType,
      featureShardId,
      size,
      dimensions,
      seed)

    val optimizationProblem = generateRandomEffectOptimizationProblem(randomEffectDataset)
    val coordinate = new RandomEffectCoordinate[SingleNodeGLMLossFunction](randomEffectDataset, optimizationProblem)
    val models = sc.parallelize(generateLinearModelsForRandomEffects(randomEffectIds, dimensions))
    val model = new RandomEffectModel(
      models,
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
  override def sparkTest(name: String)(body: => Unit): Unit =
    SparkTestUtils.SPARK_LOCAL_CONFIG.synchronized {
      sc = SparkSessionConfiguration
        .asYarnClient(new SparkConf().setMaster(SparkTestUtils.SPARK_LOCAL_CONFIG), name, useKryo = true)
        .sparkContext

      try {
        body
      } finally {
        sc.stop()
        System.clearProperty("spark.driver.port")
        System.clearProperty("spark.hostPort")
      }
    }
}
