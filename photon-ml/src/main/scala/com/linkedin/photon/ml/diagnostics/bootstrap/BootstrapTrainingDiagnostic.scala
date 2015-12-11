package com.linkedin.photon.ml.diagnostics.bootstrap

import com.linkedin.photon.ml.BootstrapTraining
import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.diagnostics.TrainingDiagnostic
import com.linkedin.photon.ml.diagnostics.reporting.reports.Utils
import com.linkedin.photon.ml.stat.BasicStatisticalSummary
import com.linkedin.photon.ml.supervised.model.{CoefficientSummary, GeneralizedLinearModel}
import org.apache.spark.rdd.RDD


class BootstrapTrainingDiagnostic(
    private val unparsedNameTermToIndex: Map[String, Int],
    private val numSamples: Int = BootstrapTrainingDiagnostic.DEFAULT_BOOTSTRAP_SAMPLES,
    private val trainingPortion: Double = BootstrapTrainingDiagnostic.DEFAULT_BOOTSTRAP_PORTION)
  extends TrainingDiagnostic[GeneralizedLinearModel, BootstrapReport] {

  import BootstrapTrainingDiagnostic._

  override def diagnose(
      modelFactory: (RDD[LabeledPoint], Map[Double, GeneralizedLinearModel]) => List[(Double, GeneralizedLinearModel)],
      models: Map[Double, GeneralizedLinearModel],
      trainingData: RDD[LabeledPoint],
      summary: Option[BasicStatisticalSummary]): Map[Double, BootstrapReport] = {

    val aggregators: Map[String, Seq[(GeneralizedLinearModel, Map[String, Double])] => Any] = Map(
      COEFFICIENTS_AGGREGATION -> BootstrapTraining.aggregateCoefficientConfidenceIntervals,
      METRICS_AGGREGATION -> BootstrapTraining.aggregateMetricsConfidenceIntervals
    )

    val aggregates = BootstrapTraining.bootstrap[GeneralizedLinearModel](
      numSamples, trainingPortion, models, modelFactory, aggregators, trainingData)

    // lambda -> metric -> metric distribution
    val metrics = aggregates.mapValues(aggregate => {
      aggregate.get(METRICS_AGGREGATION) match {
        // Should never see another case. If we do, we changed things
        // BootstrapTraining.aggregateMetricsConfidenceIntervals and so the match failure exception that we see should
        // be a clue that this here also needs to be updated.
        case Some(metricSummary: Map[String, CoefficientSummary]) => metricSummary
      }
    })

    // lambda -> coefficient summary
    val coefficients = aggregates.mapValues(aggregate => {
      aggregate.get(COEFFICIENTS_AGGREGATION) match {
        // Similar to above, but this time the change was in BootstrapTraining.aggregateCoefficientConfidenceIntervals
        case Some(coefficientSummary: (Array[CoefficientSummary], Option[CoefficientSummary])) => coefficientSummary
      }
    })

    // idx -> (name, term)
    val indexToNameTerm = unparsedNameTermToIndex.map(x => {
      val (nameTerm, index) = x
      val (name, term) = Utils.extractNameTerm(nameTerm)
      (index, (name, term))
    })

    // lambda -> (name, term) -> (index, importance, coefficient)
    val importances = coefficients.map(x => {
      val (lambda, (coeff, interept)) = x
      (lambda, coeff.zipWithIndex.map(x => {
        val (sumary, idx) = x
        val value = summary match {
          case Some(sum: BasicStatisticalSummary) =>
            sum.meanAbs(idx)
          case None =>
            1
        }
        val model = models.get(lambda)
        val c = model match {
          case Some(g: GeneralizedLinearModel) => math.abs(g.coefficients(idx))
          case _ => 1.0
        }
        val importance = value * c
        (indexToNameTerm.get(idx).get, (idx, importance, coeff(idx)))
      }).toMap)
    })

    metrics.map(item => {
      val (lambda, metricSummary) = item

      // should always have the same lambdas as keys in both these containers.
      val coefficientSummary = coefficients.get(lambda).get

      val numStraddlingZero = coefficientSummary._1
        .filter(x => x.estimateFirstQuartile() < 0 && x.estimateThirdQuartile() > 0).size

      val m = metricSummary
        .mapValues(
          x => (x.getMin(), x.estimateFirstQuartile(), x.estimateMedian(), x.estimateThirdQuartile(), x.getMax()))

      val c = coefficientSummary._1
        .map(x => (x.getMin(), x.estimateFirstQuartile(), x.estimateMedian(), x.estimateThirdQuartile(), x.getMax()))

      val straddlingZero = importances
        .getOrElse(lambda, Map.empty[(String, String), (Int, Double, CoefficientSummary)])
        .toSeq
        .filter(x => x._2._3.estimateFirstQuartile < 0 && x._2._3.estimateThirdQuartile > 0)
        .sortBy(x => x._2._2).toMap

      val importantFeatures = importances
        .get(lambda).get
        .toSeq
        .sortBy(_._2._2)
        .takeRight(NUM_IMPORTANT_FEATURES)
        .map(x => (x._1, x._2._3)).toMap

      (lambda, BootstrapReport(m, Map.empty[String, Double], importantFeatures, straddlingZero))
    })
  }
}

object BootstrapTrainingDiagnostic {
  val NUM_IMPORTANT_FEATURES = 15
  val DEFAULT_BOOTSTRAP_SAMPLES = 15
  val DEFAULT_BOOTSTRAP_PORTION = 0.7
  val COEFFICIENTS_AGGREGATION = "Coefficients"
  val METRICS_AGGREGATION = "Metrics"
  val BAGGING_AGGREGATION = "Bagging"
}
