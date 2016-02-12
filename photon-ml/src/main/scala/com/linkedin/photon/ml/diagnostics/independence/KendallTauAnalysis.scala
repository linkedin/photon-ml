package com.linkedin.photon.ml.diagnostics.independence

import org.apache.commons.math3.distribution.NormalDistribution
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

/**
 * Implement tests of independence based on [[https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient]]
 *
 */
class KendallTauAnalysis {
  import KendallTauAnalysis._

  /**
   * @param data (a, b) draws from the joint distribution P_{A,B}
   * @return analysis of independence.
   */
  def analyze(data:RDD[(Double, Double)]): KendallTauReport = {
    val count = data.count
    val rate = math.min(1.0, math.sqrt(count.toDouble)/count)
    val sampled = data.sample(false, rate, System.nanoTime()).persist(StorageLevel.MEMORY_AND_DISK_SER)
    val pairTypes = sampled.cartesian(sampled).map(x => {
      checkConcordance(x._1, x._2)
    }).aggregateByKey(0L)(seqOp = _ + _, combOp = _ + _).collect.toMap

    sampled.unpersist(false)
    analyze(pairTypes, sampled.count)
  }

  /**
   * @param data (a, b) draws from the joint distribution P_{A,B}
   * @return analysis of independence.
   */
  def analyze(data:Array[(Double, Double)]): KendallTauReport = {
    val pairTypes = (for {
      x <- 0 until data.size;
      y <- (x + 1) until data.size
    } yield {
        checkConcordance(data(x), data(y))
    }).groupBy(_._1).map( x => {
      (x._1, x._2.size.toLong)
    })

    analyze(pairTypes, data.size)
  }

  def analyze(numConcordant:Long, numDiscordant:Long, numTiesA:Long, numTiesB:Long, numItems:Long): KendallTauReport = {
    val numPairs = numItems * (numItems - 1L) / 2L
    val numNoTiesA = numPairs - numTiesA
    val numNoTiesB = numPairs - numTiesB
    val tauAlpha = (numConcordant - numDiscordant).toDouble / (numConcordant + numDiscordant).toDouble
    val tauBeta = (numConcordant - numDiscordant) / math.sqrt(numNoTiesA.toDouble * numNoTiesB.toDouble)
    val a = 2.0 * (2.0 * numItems + 5.0)
    val b = 9.0 * numItems * (numItems - 1)
    val d = if (b > 0) { math.sqrt(a / b) } else { 1.0 }
    val zAlpha = tauAlpha / d
    val pValue = GAUSSIAN_DENSITY.cumulativeProbability(math.abs(zAlpha)) -
      GAUSSIAN_DENSITY.cumulativeProbability(-math.abs(zAlpha))

    val msg = if (numTiesA + numTiesB > 0) {
      s"""
         |Note: detected ties (ties in first variable: ${numTiesA}, ties in second variable: ${numTiesB}). This means
         |that the computed z score / p value for tau-alpha over-estimates the degree of independence between A and B.
       """.stripMargin
    } else {
      ""
    }

    new KendallTauReport(
      numConcordant, numDiscordant, numItems, numPairs, numConcordant + numDiscordant,
      tauAlpha, tauBeta, zAlpha, pValue, msg)
  }

  private[this] def analyze(pairTypes:Map[String, Long], count:Long): KendallTauReport = {
    val numConcordant = pairTypes.getOrElse(CONCORDANT, 0L)
    val numDiscordant = pairTypes.getOrElse(DISCORDANT, 0L)
    val numTiesA = pairTypes.getOrElse(TIES_IN_A, 0L)
    val numTiesB = pairTypes.getOrElse(TIES_IN_B, 0L)
    analyze(numConcordant, numDiscordant, numTiesA, numTiesB, count)
  }
}

object KendallTauAnalysis {
  val CONCORDANT = "CONCORDANT"
  val DISCORDANT = "DISCORDANT"
  val TIES_IN_A = "OTHER_A"
  val TIES_IN_B = "OTHER_B"

  val GAUSSIAN_DENSITY = new NormalDistribution(0.0, 1.0)

  def checkConcordance(a:(Double, Double), b:(Double, Double)): (String, Long) = {
    val ((x1, y1), (x2, y2)) = (a, b)

    if (x1 > x2) {
      if (y1 > y2) {
        (CONCORDANT, 1L)
      } else if (y1 < y2) {
        (DISCORDANT, 1L)
      } else {
        (TIES_IN_B, 1L)
      }
    } else if (x1 < x2) {
      if (y1 < y2) {
        (CONCORDANT, 1L)
      } else if (y1 > y2) {
        (DISCORDANT, 1L)
      } else {
        (TIES_IN_B, 1L)
      }
    } else {
      (TIES_IN_A, 1L)
    }
  }
}
