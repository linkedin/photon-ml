package com.linkedin.photon.ml

import java.util.Collections

import com.google.common.base.Preconditions
import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.supervised.model.{CoefficientSummary, GeneralizedLinearModel}
import org.apache.commons.math3.distribution.UniformIntegerDistribution
import org.apache.commons.math3.random.{MersenneTwister, RandomAdaptor}
import org.apache.spark.rdd.RDD

import scala.util.Random


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
   * @return A tuple representing coefficient-wise summaries. The first part of the tuple matches 1:1 with the
   *   coefficient vector. The optional second part contains information about the intercept (if available).
   */
  def aggregateCoefficientConfidenceIntervals[GLM <: GeneralizedLinearModel](
      modelsAndMetrics: Seq[(GLM, Map[String, Double])]): (Array[CoefficientSummary], Option[CoefficientSummary]) = {

    val initState = (x: Double) => {
      val coeff = new CoefficientSummary
      coeff.accumulate(x)
      coeff
    }

    // Initialize summary state with the first model in the sequence
    val firstGLM = modelsAndMetrics.head._1
    val firstState = (
      firstGLM.coefficients.toArray.map(initState),
      firstGLM.intercept.map(initState))

    // Reduce the remaining models into a full summary
    modelsAndMetrics.tail.foldLeft(firstState)({
      case ((coeffs, intercept), (glm, _)) => {

        // Accumulate coefficients
        coeffs.zip(glm.coefficients.toArray).map({
          case (accCoeff, currCoeff) => {
            accCoeff.accumulate(currCoeff)
            accCoeff
          }
        })

        // Accumulate intercept
        intercept.map(accIntercept => {
          accIntercept.accumulate(glm.intercept.get)
          accIntercept
        })

        (coeffs, intercept)
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
   * @return A tuple representing coefficient-wise summaries. The first part of the tuple matches 1:1 with the
   *   coefficient vector. The optional second part contains information about the intercept (if available).
   */
  def aggregateMetricsConfidenceIntervals[GLM <: GeneralizedLinearModel](
      modelsAndMetrics: Seq[(GLM, Map[String, Double])]): Map[String, CoefficientSummary] = {

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
   * @param numBootstrapSamples
   * How many bootstrap samples to draw
   * @param trainModel
   * A closure that maps (RDD, warm start models) to (regularization &rarr; model) maps
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
  def bootstrap[GLM <: GeneralizedLinearModel](
      numBootstrapSamples: Int,
      populationPortionPerBootstrapSample: Double,
      warmStart: Map[Double, GLM],
      trainModel: (RDD[LabeledPoint], Map[Double, GLM]) => List[(Double, GLM)],
      aggregations: Map[String, Seq[(GLM, Map[String, Double])] => Any],
      trainingSamples: RDD[LabeledPoint]): Map[Double, Map[String, Any]] = {

    Preconditions.checkArgument(
      numBootstrapSamples > 1,
      "Number of bootstrap samples must be at least 1, got [%s]",
      numBootstrapSamples: java.lang.Integer)
    Preconditions.checkArgument(populationPortionPerBootstrapSample > 0 && populationPortionPerBootstrapSample <= 1.0,
      "Portion of training samples used for training must be in the range (0, 1.0], got [%s]",
      populationPortionPerBootstrapSample: java.lang.Double)


    val numSplits = 1000

    // regardless of what users tell us, never more than 90% for training
    val targetSplits = math.min(900, (populationPortionPerBootstrapSample * numSplits).toInt)

    val tagged = trainingSamples.mapPartitions(x => {
        val prng = new MersenneTwister(System.nanoTime())
        val dist = new UniformIntegerDistribution(prng, 0, numSplits)

        x.map(y => (dist.sample, y))
      }).cache.setName("Tagged training splits")

    val tags = (0 until numSplits).toList
    val prng = new MersenneTwister(System.nanoTime())
    (1 to numBootstrapSamples).map(x => {
      val shuffled = Random.javaRandomToRandom(RandomAdaptor.createAdaptor(prng)).shuffle(tags).toArray
      val trainTags = shuffled.slice(0, targetSplits).toSet
      val trainSet = tagged.filter(x => trainTags.contains(x._1)).map(_._2)

      val holdoutTags = shuffled.slice(targetSplits, numSplits).toSet
      val holdoutSet = tagged.filter(x => holdoutTags.contains(x._1)).map(_._2)
      val models = trainModel(trainSet, warmStart).toMap
      models.mapValues(x => (x, Evaluation.evaluate(x, holdoutSet)))
    })
      .flatMap(_.iterator)
      .toSeq
      .groupBy(_._1)
      .mapValues(x => {
        val toAggregate = x.map(_._2)
        aggregations.mapValues(_ (toAggregate))
      })
  }
}
