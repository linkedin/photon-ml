package com.linkedin.photon.ml.diagnostics.independence

import org.apache.commons.math3.random.{ISAACRandom, MersenneTwister}
import org.testng.annotations.{DataProvider, Test}


class KendallTauTest {
  import org.testng.Assert._
  import KendallTauTest._

  def printResult(result: KendallTauReport): Unit = {
   this.synchronized {
      println(
        s"""
           |Concordant pairs:      ${result.concordantPairs}
           |Discordant pairs:      ${result.discordantPairs}
           |Effective pairs:       ${result.effectivePairs},
           |Message:               ${result.messages}
           |Number of samples:     ${result.numSamples}
           |P-value:               ${result.pValueAlpha}
           |Tau alpha:             ${result.tauAlpha}
           |Tau beta:              ${result.tauBeta}
           |Total pairs:           ${result.totalPairs}
           |Z alpha:               ${result.zAlpha}
       """.stripMargin)
    }
  }

  @DataProvider
  def generateHappyPathScenarios(): Array[Array[Any]] = {
    // analyze(numConcordant:Long, numDiscordant:Long, numTiesA:Long, numTiesB:Long, numItems:Long)
    val totalPairs = NUM_SAMPLES * (NUM_SAMPLES - 1) / 2
    val p1 = totalPairs / 2
    val p2 = totalPairs - p1


    Array(
      Array("Perfectly independent", p1, p2, 0L, 0L, totalPairs, NUM_SAMPLES, 0.0, 0.0, 0.0),
      Array("Perfectly dependent", totalPairs, 0L, 0L, 0L, totalPairs, NUM_SAMPLES, 1.0, 1.0, 1.0),
      Array("Perfectly dependent 2", 0L, totalPairs, 0L, 0L, totalPairs, NUM_SAMPLES, -1.0, -1.0, 1.0),
      Array("Generates warning", p1 - 1L, p2, 1L, 0L, totalPairs, NUM_SAMPLES, 0.0, 0.0, 0.0)
    )
  }

  @Test(dataProvider = "generateHappyPathScenarios")
  def checkHappyPaths(desc:String,
                      concordant:Long,
                      discordant:Long,
                      numTiesA:Long,
                      numTiesB:Long,
                      totalPairs:Long,
                      numSamples:Long,
                      expTauAlpha:Double,
                      expTauBeta:Double,
                      expPValue:Double): Unit = {
    def analyzer = new KendallTauAnalysis
    def result = analyzer.analyze(concordant, discordant, numTiesA, numTiesB, numSamples)

    printResult(result)
    assertEquals(result.pValueAlpha, expPValue, P_VALUE_TOLERANCE)
    assertEquals(result.concordantPairs, concordant)
    assertEquals(result.discordantPairs, discordant)
    assertEquals(result.totalPairs, totalPairs)
    assertEquals(result.tauAlpha, expTauAlpha, TAU_ALPHA_TOLERANCE)
    assertEquals(result.tauBeta, expTauBeta, TAU_BETA_TOLERANCE)
    assertEquals(result.effectivePairs, concordant + discordant)
    if (numTiesA + numTiesB > 0L) {
      assertNotEquals(result.messages, "")
    }
  }

  @Test
  def checkSampledDependentData(): Unit = {
    val prng = new ISAACRandom(0xdeadbeef)
    def samples = (0L until NUM_SAMPLES).map(x => {
      val y = prng.nextDouble
      (y, y*y)
    }).toArray
    def analyzer = new KendallTauAnalysis
    def result = analyzer.analyze(samples)
    printResult(result)
    assertEquals(result.pValueAlpha, 1.0, P_VALUE_TOLERANCE)
    assertEquals(result.concordantPairs, result.totalPairs)
    assertEquals(result.totalPairs, result.effectivePairs)
    assertEquals(result.discordantPairs, 0L)
    assertEquals(result.effectivePairs, NUM_SAMPLES * (NUM_SAMPLES - 1)/2)
    assertEquals(result.tauAlpha, 1.0, TAU_ALPHA_TOLERANCE)
    assertEquals(result.tauBeta, 1.0, TAU_BETA_TOLERANCE)
  }

}

object KendallTauTest {
  val NUM_SAMPLES = 100L
  val P_VALUE_TOLERANCE = 0.10
  val TAU_ALPHA_TOLERANCE = 1e-3
  val TAU_BETA_TOLERANCE = 1e-3
}
