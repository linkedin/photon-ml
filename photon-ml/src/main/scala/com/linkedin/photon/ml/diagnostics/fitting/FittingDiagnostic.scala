package com.linkedin.photon.ml.diagnostics.fitting

import java.util.concurrent.Executors

import com.google.common.base.Preconditions
import com.linkedin.photon.ml.Evaluation
import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.diagnostics.TrainingDiagnostic
import com.linkedin.photon.ml.stat.BasicStatisticalSummary
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import org.apache.spark.rdd.RDD

import scala.concurrent.duration.Duration
import scala.concurrent.{Await, Future, ExecutionContext}

/**
 * Try to diagnose under/over-fit and label bias problems in a data set. The idea here is that by computing metrics
 * on both a training set and a hold-out set, we can show how the metrics move as a function of the training set size
 *
 * @param numConcurrentFits
 *                          Number of model trainings to launch in parallel
 */
class FittingDiagnostic(numConcurrentFits:Int=4) extends TrainingDiagnostic[GeneralizedLinearModel, FittingReport] {
  import FittingDiagnostic._

  Preconditions.checkArgument(numConcurrentFits > 0 && numConcurrentFits <= MAX_CONCURRENT_FITS,
                              s"Number of concurrent fits [%s] must be in the range (0, $MAX_CONCURRENT_FITS]",
                              numConcurrentFits : java.lang.Integer)

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
  def diagnose(modelFactory:RDD[LabeledPoint]=>List[(Double, GeneralizedLinearModel)], trainingSet:RDD[LabeledPoint], summary:Option[BasicStatisticalSummary]): Map[Double, FittingReport] = {
    val dimension:Long = trainingSet.first.features.length
    val numSamples = trainingSet.count
    val minSamples = dimension * MIN_SAMPLES_PER_PARTITION_PER_DIMENSION

    if (numSamples > minSamples) {
      val ec = ExecutionContext.fromExecutor(Executors.newFixedThreadPool(numConcurrentFits))

      val splits = trainingSet.randomSplit((0 to NUM_TRAINING_PARTITIONS).map(x => 1.0 / NUM_TRAINING_PARTITIONS).toArray, System.nanoTime())
      val holdOut = splits(NUM_TRAINING_PARTITIONS - 1)

      (for (subset <- 1 until NUM_TRAINING_PARTITIONS) yield {
        // At some future point, we will have fit a model
        Future {
          val dataPortion = subset.toDouble / NUM_TRAINING_PARTITIONS
          val dataSet = (0 until subset).map(x => splits(x)).fold(trainingSet.sparkContext.emptyRDD[LabeledPoint])((x, y) => x.union(y))
          val models = modelFactory(dataSet)
          val metricsTest = models.map(x => (x._1, Evaluation.evaluate(x._2, holdOut))).toMap
          val metricsTrain = models.map(x => (x._1, Evaluation.evaluate(x._2, dataSet))).toMap
          (dataPortion, metricsTest, metricsTrain)
        } (ec)
      })
        // Now wait for the parallel fits to finish
      .map(Await.result(_, Duration.Inf))
        // Convert into (lambda, metric, portion, test, train) tuples
      .flatMap(x => {
        val (portion, testMetrics, trainMetrics) = x

        (for { lambdaTestMetrics <- testMetrics} yield {
          val (lambda, test) = lambdaTestMetrics
          val train = trainMetrics.getOrElse(lambda, Map.empty)
          for { metricTypeTestValue <- test } yield {
            val (metric, testValue) = metricTypeTestValue
            (lambda, metric, portion, testValue, train.get(metric).get)
          }
        })
      })
      .flatMap(_.iterator)
        // Gather by lambda
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
    } else {
      Map.empty
    }
  }
}

object FittingDiagnostic {
  def MAX_CONCURRENT_FITS = 4
  def NUM_TRAINING_PARTITIONS = 10
  def MIN_SAMPLES_PER_PARTITION_PER_DIMENSION = 100
}