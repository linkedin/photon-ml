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
package com.linkedin.photon.ml.diagnostics.fitting

import org.apache.commons.math3.distribution.UniformIntegerDistribution
import org.apache.commons.math3.random.MersenneTwister
import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.Evaluation
import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.diagnostics.TrainingDiagnostic
import com.linkedin.photon.ml.stat.FeatureDataStatistics
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import com.linkedin.photon.ml.util.Logging

/**
 * Try to diagnose under/over-fit and label bias problems in a dataset. The idea here is that by computing metrics
 * on both a training set and a hold-out set, we can show how the metrics move as a function of the training set size
 *
 */
class FittingDiagnostic extends TrainingDiagnostic[GeneralizedLinearModel, FittingReport] with Logging {

  import FittingDiagnostic._

  /**
   *
   * @param modelFactory
   * Functor that, given a dataset, produces a set of (lambda, model) tuples
   *
   * @param trainingSet
   * @param summary
   * @return
   * A map of (lambda &rarr; fitting report). If there is not enough information to produce a reasonable report, we
   * return an empty map.
   */
  def diagnose(
      modelFactory: (RDD[LabeledPoint], Map[Double, GeneralizedLinearModel]) => List[(Double, GeneralizedLinearModel)],
      warmStart: Map[Double, GeneralizedLinearModel],
      trainingSet: RDD[LabeledPoint],
      summary: Option[FeatureDataStatistics],
      seed: Long = System.nanoTime): Map[Double, FittingReport] = {

    val numSamples = trainingSet.count()
    val dimension = trainingSet.first().features.size
    val minSamples = dimension * MIN_SAMPLES_PER_PARTITION_PER_DIMENSION

    if (numSamples > minSamples) {
      val tagged = trainingSet.mapPartitions(partition => {
        val prng = new MersenneTwister(seed)
        val dist = new UniformIntegerDistribution(prng, 0, NUM_TRAINING_PARTITIONS)
        partition.map(x => (dist.sample(), x))
      }).cache().setName("Tagged samples for learning curves")

      val holdOut = tagged
        .filter(_._1 == NUM_TRAINING_PARTITIONS - 1)
        .map(_._2)
        .cache()
        .setName("Hold out for learning curves")

      val result = (0 until (NUM_TRAINING_PARTITIONS - 1))
        .scanLeft(
          (0.0, warmStart, Map[Double, Map[String, Double]](), Map[Double, Map[String, Double]]()))( (prev, maxTag) => {

        val dataset = tagged.filter(_._1 <= maxTag).map(_._2)
        val startTime = System.currentTimeMillis
        val samples = dataset.count()
        val dataPortion = 100.0 * samples / numSamples
        logger.info(s"Data portion: $dataPortion ==> warm start models with lambdas = ${warmStart.keys.mkString(", ")}")
        val models = modelFactory(dataset, prev._2).toMap

        val metricsTest = models.mapValues(x => Evaluation.evaluate(x, holdOut))
        val metricsTrain = models.mapValues(x => Evaluation.evaluate(x, dataset))
        dataset.unpersist(blocking = false)
        val elapsedTime = (System.currentTimeMillis - startTime) / 1000.0
        logger.info(s"Training on $dataPortion%% of the data took $elapsedTime seconds")

        (dataPortion, models, metricsTest, metricsTrain)
      })
      .map(x => (x._1, x._3, x._4))
      .flatMap(x => {
      val (portion, testMetrics, trainMetrics) = x
      (for { lambdaTestMetrics <- testMetrics} yield {
        val (lambda, test) = lambdaTestMetrics
        val train = trainMetrics.getOrElse(lambda, Map.empty)
        for { metricTypeTestValue <- test } yield {
          val (metric, testValue) = metricTypeTestValue
          (lambda, metric, portion, testValue, train.get(metric).get)
        }
      }).iterator
    })
    .flatMap(_.iterator)
    .groupBy(_._1)
      .map(x => {
        val (lambda, tuplesByLambda) = x
        val byMetric = tuplesByLambda.map(x => (x._2, x._3, x._4, x._5)).groupBy(_._1).map(x => {
          val (metric, data) = x
          val sorted = data.sortBy(_._2)
          val portions = sorted.map(_._2).toArray
          val test = sorted.map(_._3).toArray
          val train = sorted.map(_._4).toArray
          (metric, (portions, train, test))
        })
        (lambda, new FittingReport(byMetric, ""))
      })


      holdOut.unpersist(blocking = false)
      tagged.unpersist(blocking = false)
      result
    } else {
      Map.empty
    }
  }
}

object FittingDiagnostic {
  val NUM_TRAINING_PARTITIONS = 10
  val MIN_SAMPLES_PER_PARTITION_PER_DIMENSION = 10
}
