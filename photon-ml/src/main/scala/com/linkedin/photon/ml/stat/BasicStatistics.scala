package com.linkedin.photon.ml.stat

import breeze.linalg.DenseVector
import breeze.linalg.Vector
import com.linkedin.photon.ml.data
import com.linkedin.photon.ml.data.LabeledPoint
import org.apache.spark.mllib.linalg.VectorsWrapper
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.rdd.RDD


/**
 * An object to generate basic statistics (e.g., max, min, mean, variance) of [[data.LabeledPoint]] RDD.
 *
 * @author dpeng
 */
private[ml] object BasicStatistics {

  /**
   * Generate basic statistics (e.g., max, min, mean, variance) of [[data.LabeledPoint]] RDD using the mllib
   * interface.
   * @param inputData Input data as [[data.LabeledPoint]] RDD
   *
   */
  def getBasicStatistics(inputData: RDD[LabeledPoint]): BasicStatisticalSummary = {
    val mllibSummary = Statistics.colStats(inputData.map(x => VectorsWrapper.breezeToMllib(x.features)))

    val length = inputData.take(1)(0).features.length
    val zero = (0L, DenseVector.zeros[Long](length), DenseVector.zeros[Double](length))

    val (totalSamples, nonzeroCount, nonzeroMean) = inputData.mapPartitions(x => {
      if (x.hasNext) {
        var count = 1L
        val one = x.take(1).next()
        val sums:Vector[Double] = one.features.copy
        val counts:Vector[Long] = sums.mapValues(x => if (math.abs(x) > 1e-12) 1L else 0L).toDenseVector

        x.foreach(y => {
          count += 1
          y.features.iterator.foreach(f => {
            val q = math.abs(f._2)
            counts(f._1) += 1
            sums(f._1) += (q - sums(f._1))/counts(f._1)
            (f._1, 0.0)
          })
        })

        Seq((count, counts, sums)).iterator
      } else {
        Seq.empty.iterator
      }
    }).fold(zero)(op = (x, y) => {
      val totalCount = x._1 + y._1
      val newCounts = x._2 + y._2
      val delta = y._3 - x._3
      val scales = y._2.mapValues(x => x.toDouble) :/ newCounts.mapValues(x => if (x != 0) x.toDouble else 1.0)
      val newMeans = x._3 + delta :* scales
      (totalCount, newCounts, newMeans)
    })

    var meanAbs:Vector[Double] = DenseVector.zeros[Double](length)

    if (totalSamples > 0L) {
      meanAbs = nonzeroMean :* (totalSamples - nonzeroCount).mapValues(x => 1.0 - x.toDouble / totalSamples)
    }

    BasicStatisticalSummary(mllibSummary, meanAbs)
  }

}
