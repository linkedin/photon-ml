package com.linkedin.photon.ml.diagnostics.fitting

import com.linkedin.photon.ml.Evaluation
import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.diagnostics.TrainingDiagnostic
import com.linkedin.photon.ml.stat.BasicStatisticalSummary
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import org.apache.commons.math3.distribution.UniformIntegerDistribution
import org.apache.commons.math3.random.MersenneTwister
import org.apache.spark.Logging
import org.apache.spark.rdd.RDD


/**
 * Try to diagnose under/over-fit and label bias problems in a data set. The idea here is that by computing metrics
 * on both a training set and a hold-out set, we can show how the metrics move as a function of the training set size
 *
 */
class FittingDiagnostic extends TrainingDiagnostic[GeneralizedLinearModel, FittingReport] with Logging {

  import FittingDiagnostic._

  /**
   *
   * @param modelFactory
   * Functor that, given a data set, produces a set of (lambda, model) tuples
   *
   * @param trainingSet
   * @param summary
   * @return
   * A map of (lambda &rarr; fitting report). If there is not enough information to produce a reasonable report, we
   * return an empty map.
   */
  def diagnose(modelFactory: (RDD[LabeledPoint], Map[Double, GeneralizedLinearModel]) => List[(Double, GeneralizedLinearModel)],
               warmStart: Map[Double, GeneralizedLinearModel],
               trainingSet: RDD[LabeledPoint], summary: Option[BasicStatisticalSummary]): Map[Double, FittingReport] = {
    val numSamples = trainingSet.count
    val dimension = trainingSet.first.features.size
    val minSamples = dimension * MIN_SAMPLES_PER_PARTITION_PER_DIMENSION

    if (numSamples > minSamples) {
      val tagged = trainingSet.mapPartitions(partition => {
        val prng = new MersenneTwister(System.nanoTime)
        val dist = new UniformIntegerDistribution(prng, 0, NUM_TRAINING_PARTITIONS)
        partition.map(x => (dist.sample(), x))
      }).cache.setName("Tagged samples for learning curves")

      val holdOut = tagged.filter(_._1 == NUM_TRAINING_PARTITIONS - 1).map(_._2).cache.setName("Hold out for learning curves")

      val result = (0 until (NUM_TRAINING_PARTITIONS - 1)).scanLeft((0.0, warmStart, Map[Double, Map[String, Double]](), Map[Double, Map[String, Double]]()))( (prev, maxTag) => {
        val dataSet = tagged.filter(_._1 <= maxTag).map(_._2)
        val startTime = System.currentTimeMillis
        val samples = dataSet.count
        val dataPortion = 100.0 * samples / numSamples
        logInfo(s"Data portion: ${dataPortion} ==> warm start models with lambdas = ${warmStart.keys.mkString(", ")}")
        val models = modelFactory(dataSet, prev._2).toMap

        val metricsTest = models.mapValues(x => Evaluation.evaluate(x, holdOut))
        val metricsTrain = models.mapValues(x => Evaluation.evaluate(x, dataSet))
        dataSet.unpersist(false)
        val elapsedTime = (System.currentTimeMillis - startTime) / 1000.0
        logInfo(s"Training on $dataPortion%% of the data took $elapsedTime seconds")

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


      holdOut.unpersist(false)
      tagged.unpersist(false)
      result
    } else {
      Map.empty
    }
  }
}

object FittingDiagnostic {
  def NUM_TRAINING_PARTITIONS = 10
  def MIN_SAMPLES_PER_PARTITION_PER_DIMENSION = 100
}