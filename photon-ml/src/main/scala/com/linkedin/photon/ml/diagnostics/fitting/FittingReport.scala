package com.linkedin.photon.ml.diagnostics.fitting

import com.linkedin.photon.ml.diagnostics.reporting.LogicalReport

/**
 * Demonstrate how model metrics change as a function of the volume of data used to fit the model, both on the training
 * set and a held-out set
 *
 * @param metrics Map of (metric name &rarr; (% training set, performance on training set, performance on held out)
 *   tuples
 *
 * @param fittingMsg Description of any questions / comments / concerns that came up while testing how well we fit
 */
case class FittingReport(val metrics:Map[String, (Array[Double], Array[Double], Array[Double])],
                         val fittingMsg:String) extends LogicalReport
