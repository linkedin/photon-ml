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
package com.linkedin.photon.ml.optimization

import scala.collection.mutable
import scala.math.abs
import scala.reflect.ClassTag

import breeze.linalg.{Vector, sum}
import org.apache.spark.Logging
import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.data.{LabeledPoint, ObjectProvider}
import com.linkedin.photon.ml.function.DiffFunction
import com.linkedin.photon.ml.normalization.NormalizationContext
import com.linkedin.photon.ml.sampler.DownSampler
import com.linkedin.photon.ml.supervised.model.{GeneralizedLinearModel, ModelTracker}

/**
  * GeneralizedOptimizationProblem implements methods to train a Generalized Linear Model (GLM).
  * This class should be extended with a loss function and the createModel function to create a new GLM.
  *
  * @param optimizer The underlying optimizer who does the job
  * @param objectiveFunction The objective function upon which to optimize
  * @param regularizationWeight The regularization weight of the optimization problem
  * @param sampler The sampler used to down-sample the training data points
  * @tparam GLM The type of returned generalized linear model
  * @tparam F The type of loss function of the generalized linear algorithm
  */
abstract class GeneralizedLinearOptimizationProblem[+GLM <: GeneralizedLinearModel : ClassTag,
  // TODO: covariance here is temporary measure -- let's revisit this
  +F <: DiffFunction[LabeledPoint]](
    optimizer: Optimizer[LabeledPoint, F],
    objectiveFunction: F,
    sampler: DownSampler,
    regularizationContext: RegularizationContext,
    regularizationWeight: Double,
    modelTrackerBuilder: Option[mutable.ListBuffer[ModelTracker]],
    treeAggregateDepth: Int,
    isComputingVariances: Boolean) extends Logging with Serializable {

   def downSample(labeledPoints: RDD[(Long, LabeledPoint)]): RDD[(Long, LabeledPoint)] = {
     sampler.downSample(labeledPoints)
   }

  /**
    * Get the optimization state trackers for the optimization problems solved
    *
    * @return Some(OptimizationStatesTracker) if one was kept, otherwise None
    */
  def getStatesTracker: Option[OptimizationStatesTracker] = optimizer.getStateTracker

  /**
    * Get models for the intermediate optimization states of the optimization problems solved
    *
    * @return Some(list of ModelTrackers) if ModelTrackers were being kept, otherwise None
    */
  def getModelTracker: Option[List[ModelTracker]] = modelTrackerBuilder.map(_.toList)

  /**
    * Updates properties of the objective function. Useful in cases of data-related changes or parameter sweep.
    *
    * @param normalizationContext new normalization context
    * @param regularizationWeight new regulariation weight
    * @return a new optimization problem with updated objective
    */
  def updateObjective(
      normalizationContext: ObjectProvider[NormalizationContext],
      regularizationWeight: Double): GeneralizedLinearOptimizationProblem[GLM, F]

  /**
    * Create a default generalized linear model with 0-valued coefficients
    *
    * @param dimension The dimensionality of the model coefficients
    * @return A model with zero coefficients
    */
  def initializeZeroModel(dimension: Int): GLM

  /**
    * Create a model given the coefficients
    *
    * @param coefficients The coefficients parameter of each feature (and potentially including intercept)
    * @param variances The coefficient variances
    * @return A generalized linear model with coefficients parameters
    */
  protected def createModel(coefficients: Vector[Double], variances: Option[Vector[Double]]): GLM

  /**
    * Create a model given the coefficients
    *
    * @param normalizationContext The normalization context
    * @param coefficients A vector of feature coefficients (and potentially including intercept)
    * @param variances The coefficient variances
    * @return A generalized linear model with intercept and coefficients parameters
    */
  protected def createModel(
    normalizationContext: NormalizationContext,
    coefficients: Vector[Double],
    variances: Option[Vector[Double]]): GLM = {

    createModel(
      normalizationContext.transformModelCoefficients(coefficients),
      variances.map(normalizationContext.transformModelCoefficients))
  }

  /**
    * Compute coefficient variances
    *
    * @param labeledPoints The training dataset
    * @param coefficients The model coefficients
    * @return The coefficient variances
    */
  protected def computeVariances(labeledPoints: RDD[LabeledPoint], coefficients: Vector[Double]): Option[Vector[Double]]

  /**
    * Compute coefficient variances
    *
    * @param labeledPoints The training dataset
    * @param coefficients The model coefficients
    * @return The coefficient variances
    */
  protected def computeVariances(labeledPoints: Iterable[LabeledPoint], coefficients: Vector[Double])
    : Option[Vector[Double]]

  /**
    * Run the algorithm with the configured parameters on an input RDD of LabeledPoint entries.
    *
    * @param input A RDD of input labeled data points in the original scale
    * @param normalizationContext The normalization context
    * @return The learned generalized linear models of each regularization weight and iteration.
    */
  def run(input: RDD[LabeledPoint], normalizationContext: NormalizationContext): GLM = {
    val numFeatures = input.first().features.size
    val initialWeight = Vector.zeros[Double](numFeatures)
    val initialModel = createModel(initialWeight, None)

    run(input, initialModel, normalizationContext)
  }

  /**
    * Run the algorithm with the configured parameters on an input RDD of LabeledPoint entries
    * starting from the initial model provided.
    *
    * @param input A RDD of input labeled data points in the normalized scale (if normalization is enabled)
    * @param initialModel The initial model
    * @param normalizationContext The normalization context
    * @return The learned generalized linear models of each regularization weight and iteration.
    */
  def run(
      input: RDD[LabeledPoint],
      initialModel: GeneralizedLinearModel,
      normalizationContext: NormalizationContext): GLM = {

    val (optimizedCoefficients, _) = optimizer.optimize(input, objectiveFunction, initialModel.coefficients.means)
    val optimizedVariances = computeVariances(input, optimizedCoefficients)

    modelTrackerBuilder.foreach { modelTrackerBuilder =>
      val tracker = optimizer.getStateTracker.get
      logInfo(s"History tracker information:\n $tracker")
      val modelsPerIteration = tracker.getTrackedStates.map { x =>
        val coefficients = x.coefficients
        val variances = computeVariances(input, coefficients)
        createModel(normalizationContext, coefficients, variances)
      }
      logInfo(s"Number of iterations: ${modelsPerIteration.length}")
      modelTrackerBuilder += new ModelTracker(tracker.toString, modelsPerIteration)
    }

    createModel(normalizationContext, optimizedCoefficients, optimizedVariances)
  }

  /**
    * Run the algorithm with the configured parameters on an input RDD of LabeledPoint entries
    * starting from the initial model provided.
    *
    * @param input A RDD of input labeled data points in the normalized scale (if normalization is enabled)
    * @param initialModel The initial model
    * @param normalizationContext The normalization context
    * @return The learned generalized linear models of each regularization weight and iteration.
    */
  def run(
      input: Iterable[LabeledPoint],
      initialModel: GeneralizedLinearModel,
      normalizationContext: NormalizationContext): GLM = {

    val (optimizedCoefficients, _) = optimizer.optimize(input, objectiveFunction, initialModel.coefficients.means)
    val optimizedVariances = computeVariances(input, optimizedCoefficients)

    modelTrackerBuilder.foreach { modelTrackerBuilder =>
      val tracker = optimizer.getStateTracker.get
      logInfo(s"History tracker information:\n $tracker")
      val modelsPerIteration = tracker.getTrackedStates.map { x =>
        val coefficients = x.coefficients
        val variances = computeVariances(input, coefficients)
        createModel(normalizationContext, coefficients, variances)
      }
      logInfo(s"Number of iterations: ${modelsPerIteration.length}")
      modelTrackerBuilder += new ModelTracker(tracker.toString, modelsPerIteration)
    }

    createModel(normalizationContext, optimizedCoefficients, optimizedVariances)
  }

  /**
    * Compute the regularization term value
    *
    * @param model the model
    * @return regularization term value
    */
  def getRegularizationTermValue(model: GeneralizedLinearModel): Double = {
    import GeneralizedLinearOptimizationProblem._

    regularizationContext.regularizationType match {
      case RegularizationType.L1 => getL1RegularizationTermValue(model, regularizationWeight)
      case RegularizationType.L2 => getL2RegularizationTermValue(model, regularizationWeight)
      case RegularizationType.ELASTIC_NET => getElasticNetRegularizationTermValue(model, regularizationWeight,
        regularizationContext)
      case _ => 0.0
    }
  }
}

object GeneralizedLinearOptimizationProblem {
  /**
    * Compute the L1 regularization term value
    *
    * @param model the model
    * @param regularizationWeight the weight of the regularization value
    * @return L1 regularization term value
    */
  protected[ml] def getL1RegularizationTermValue(model: GeneralizedLinearModel, regularizationWeight: Double)
    : Double = {

    val coefficients = model.coefficients.means
    sum(coefficients.map(abs)) * regularizationWeight
  }

  /**
    * Compute the L2 regularization term value
    *
    * @param model the model
    * @param regularizationWeight the weight of the regularization value
    * @return L2 regularization term value
    */
  protected[ml] def getL2RegularizationTermValue(model: GeneralizedLinearModel, regularizationWeight: Double)
    : Double = {

    val coefficients = model.coefficients.means
    coefficients.dot(coefficients) * regularizationWeight / 2
  }

  /**
    * Compute the Elastic Net regularization term value
    *
    * @param model the model
    * @param regularizationWeight the weight of the regularization value
    * @param regularizationContext the regularization context
    * @return Elastic Net regularization term value
    */
  protected[ml] def getElasticNetRegularizationTermValue(model: GeneralizedLinearModel, regularizationWeight: Double,
      regularizationContext: RegularizationContext): Double = {

    val (l1weight, l2weight) = (regularizationContext.getL1RegularizationWeight(regularizationWeight),
      regularizationContext.getL2RegularizationWeight(regularizationWeight))
    getL1RegularizationTermValue(model, l1weight) + getL2RegularizationTermValue(model, l2weight)
  }
}
