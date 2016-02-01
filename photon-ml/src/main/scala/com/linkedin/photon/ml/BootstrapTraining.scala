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

import java.util.concurrent.Executors

import com.google.common.base.Preconditions
import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.supervised.model.{CoefficientSummary, GeneralizedLinearModel}
import org.apache.spark.rdd.RDD

import scala.concurrent.duration.Duration
import scala.concurrent.{Await, Future, ExecutionContext}

/**
 * General framework for bootstrap training.
 */
object BootstrapTraining {
  /**
   * Aggregate coefficient-wise statistics for a set of models as a way of estimating confidence intervals around
   * model coefficients. The intent is that this information can later be consumed in a variety of ways:
   *
   * <ul>
   * <li>It provides useful diagnostic information about which coefficients are "significant" (i.e. very unlikely to
   * be zero)</li>
   * <li>It helps understand uncertainty in the model (wide intervals for a particular coefficient imply that there
   * may be some feature engineering that could be done with the corresponding feature to improve performance)</li>
   * <li>The information could be consumed by downstream components. E.g: post-modeling feature selection (drop all
   * features whose CI includes zero; drop all features whose coefficient has too large variance, etc.)</li>
   * </ul>
   * @param modelsAndMetrics
   * @tparam GLM
   * @return A tuple representing coefficient-wise summaries. The first part of the tuple matches 1:1 with the coefficient
   *         vector. The optional second part contains information about the intercept (if available).
   */
  def aggregateCoefficientConfidenceIntervals[GLM <: GeneralizedLinearModel](modelsAndMetrics: Seq[(GLM, Map[String, Double])]): (Array[CoefficientSummary], Option[CoefficientSummary]) = {
    var resultCoeff: Array[CoefficientSummary] = null
    var resultIntercept: Option[CoefficientSummary] = null
    val initState = (x: Double) => {
      val coeff = new CoefficientSummary
      coeff.accumulate(x)
      coeff
    }

    modelsAndMetrics.foldLeft(null: (Array[CoefficientSummary], Option[CoefficientSummary]))((state, sample) => {
      if (null == state) {
        (sample._1.coefficients.toArray.map(initState(_)),
          sample._1.intercept.map(initState(_)))
      } else {
        (state._1.zip(sample._1.coefficients.toArray).map(x => {
          x._1.accumulate(x._2)
          x._1
        }),
          state._2.map(x => {
            x.accumulate(sample._1.intercept.get)
            x
          }))
      }
    })
  }

  /**
   * Aggregate coefficient-wise statistics for a set of models as a way of estimating confidence intervals around
   * model coefficients. The intent is that this information can later be consumed in a variety of ways:
   *
   * <ul>
   * <li>It provides useful diagnostic information about which coefficients are "significant" (i.e. very unlikely to
   * be zero)</li>
   * <li>It helps understand uncertainty in the model (wide intervals for a particular coefficient imply that there
   * may be some feature engineering that could be done with the corresponding feature to improve performance)</li>
   * <li>The information could be consumed by downstream components. E.g: post-modeling feature selection (drop all
   * features whose CI includes zero; drop all features whose coefficient has too large variance, etc.)</li>
   * </ul>
   * @param modelsAndMetrics
   * @tparam GLM
   * @return A tuple representing coefficient-wise summaries. The first part of the tuple matches 1:1 with the coefficient
   *         vector. The optional second part contains information about the intercept (if available).
   */
  def aggregateMetricsConfidenceIntervals[GLM <: GeneralizedLinearModel](modelsAndMetrics: Seq[(GLM, Map[String, Double])]): Map[String, CoefficientSummary] = {
    modelsAndMetrics.map(_._2).flatMap(_.iterator).groupBy(_._1).mapValues(x => {
      val values = x.map(_._2)
      values.foldLeft(new CoefficientSummary)((sum, sample) => {
        sum.accumulate(sample)
        sum
      })
    })
  }

  /**
   * Implement bootstrapping. This works as follows:
   * <ul>
   * <li>Repeat <tt>numBootstrapSamples</tt> times:</li>
   * <ul>
   * <li>Draw a random sample of <tt>trainingSamples</tt> (sampling with replacement, uniform probability)</li>
   * <li>Use <tt>trainModel</tt> (likely a curried call to ModelTraining.trainGeneralizedLinearModel) to
   * produce a map of (regularization &rarr; model) tuples.</li>
   * </ul>
   * <li>Use the samples to produce a map of (regularization &rarr; aggregationType &rarr; aggregate)</li>
   * </ul>
   *
   * The choice of aggregations will determine what the output of this function is. For example, using
   * [[BootstrapTraining.aggregateCoefficientConfidenceIntervals()]] will produce useful diagnostic metadata about
   * a model trained on the full data set, while one could just as easily use an aggregator that binds the underlying
   * models to a bootstrap aggregating adapter to get model bagging.
   *
   * @param concurrency
   * Maximum number of bootstrap samples to draw concurrently
   * @param numBootstrapSamples
   * How many bootstrap samples to draw
   * @param seed
   * Base PRNG seed used for drawing samples. For sample i, we use seed + i as the actual seed to avoid
   * producing duplicate draws
   * @param trainModel
   * A closure that maps RDDs to (regularization &rarr; model) maps
   * @param aggregations
   * A map of (name &rarr; closure that implements aggregation operation)
   * @param trainingSamples
   * RDD used for training
   * @param populationPortionPerBootstrapSample
   * Fraction in the range of (0, 1) of trainingSamples to use when fitting a bootstrap model. The remaining fraction is
   * used as a hold-out to evaluate.
   * @tparam GLM
   * Model type
   * @return
   * A two level map of (regularization &rarr; aggregate name &rarr; aggregate) computed aggregates,
   * broken down by regularization.
   */
  def bootstrap[GLM <: GeneralizedLinearModel](concurrency: Int,
                                               numBootstrapSamples: Int,
                                               seed: Long,
                                               populationPortionPerBootstrapSample: Double,
                                               trainModel: RDD[LabeledPoint] => Map[Double, GLM],
                                               aggregations: Map[String, Seq[(GLM, Map[String, Double])] => Any],
                                               trainingSamples: RDD[LabeledPoint]): Map[Double, Map[String, Any]] = {

    Preconditions.checkArgument(
      concurrency > 1,
      "Concurrency must be at least 1, got %s", concurrency: java.lang.Integer)
    Preconditions.checkArgument(
      numBootstrapSamples > 1,
      "Number of bootstrap samples must be at least 1, got [%s]",
      numBootstrapSamples: java.lang.Integer)
    Preconditions.checkArgument(populationPortionPerBootstrapSample > 0 && populationPortionPerBootstrapSample <= 1.0,
      "Portion of training samples used for training must be in the range (0, 1.0], got [%s]",
      populationPortionPerBootstrapSample: java.lang.Double)

    implicit val execContext = ExecutionContext.fromExecutor(Executors.newFixedThreadPool(concurrency))

    // Train a model for each sample
    Await.result(Future.traverse(1 to numBootstrapSamples)(x => Future {
      val split = trainingSamples.randomSplit(Array(0.7, 0.3), System.nanoTime)
      val models = trainModel(split(0))
      models.mapValues(x => (x, Evaluation.evaluate(x, split(1))))
    }(execContext)), Duration.Inf)
      .flatMap(_.iterator)
      .toSeq
      .groupBy(_._1)
      .mapValues(x => {
        val toAggregate = x.map(_._2)
        aggregations.mapValues(_ (toAggregate))
      })
  }
}
