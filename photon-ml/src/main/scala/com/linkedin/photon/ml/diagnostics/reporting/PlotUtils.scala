package com.linkedin.photon.ml.diagnostics.reporting

import com.linkedin.photon.ml.Evaluation

/**
 * Constants and other utility methods for handling plots
 */
object PlotUtils {
  val PLOT_HEIGHT = 960
  val PLOT_WIDTH = 1280
  val EPSILON = 1e-9

  /**
   * Handle the specific case of figuring out a valid range for plotting a metric.
   *
   * @param metric name of the metric being plotted.
   * @param values values being plotted
   * @return
   *         If the metric has a known range, return that range; otherwise, compute a
   *         reasonable range from the data.
   */
  def getRangeForMetric(metric: String, values: Seq[Double]): (Double, Double) = {

    val metadata = Evaluation.metricMetadata.get(metric)
    require(metadata.isDefined, s"Require Evaluation.metricMetadata entry for metric [$metric]")

    metadata.get.rangeOption match {
      case Some(x) => x
      case None =>
        getRange(values)
    }
  }

  /**
   * Compute a reasonable range for an axis given the set of values to be plotted
   * @param values
   */
  def getRange(values: Seq[Double]): (Double, Double) = {
    val min = values.min
    val max = values.max
    val mid = (max + min) / 2.0
    val diff = math.abs(max - min)


    require(!mid.isInfinite && !mid.isNaN, s"Computed midpoint of ${values.mkString(", ")} must be finite")

    if (math.abs(max - min) < EPSILON) (0.99 * mid - EPSILON, 1.01 * mid + EPSILON) else (min - 0.01 * diff, max + 0.01 * diff)
  }
}
