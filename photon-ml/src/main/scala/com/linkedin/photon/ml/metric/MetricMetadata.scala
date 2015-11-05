package com.linkedin.photon.ml.metric

/**
 * Metadata about a particular metric.
 * @param name Metric name
 * @param description Metric description
 * @param worstToBestOrdering Ordering that can be used to sort from worst to best
 * @param rangeOption If present, tuple of (minValue, maxValue)
 */
case class MetricMetadata(name:String,
                          description:String,
                          worstToBestOrdering:Ordering[Double],
                          rangeOption:Option[(Double, Double)])
