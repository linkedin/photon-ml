package com.linkedin.photon.ml.diagnostics.fitting

import com.linkedin.photon.ml.diagnostics.reporting.SectionPhysicalReport
import org.testng.annotations.Test

/**
 * Sanity checks for FittingToPhysicalReportTransformer
 */
class FittingToPhysicalReportTransformerTest {
  import FittingToPhysicalReportTransformerTest._
  import org.testng.Assert._

  @Test
  def checkHappyPath(): Unit = {
    val xData = (1 until NUM_SAMPLES).map( x => x.toDouble / NUM_SAMPLES ).toArray
    val report = new FittingReport(
      Map(FIRST_METRIC -> (xData, xData, xData),
          SECOND_METRIC -> (xData, xData, xData)),
      "Message"
    )
    val transformer = new FittingToPhysicalReportTransformer()
    val transformed = transformer.transform(report)

    assertTrue(transformed.isInstanceOf[SectionPhysicalReport])
    assertEquals(transformed.items.length, EXPECTED_SECTIONS)
    transformed.items.map( x => {
      assertTrue(x.isInstanceOf[SectionPhysicalReport])
    })
  }
}

object FittingToPhysicalReportTransformerTest {
  val FIRST_METRIC = "FIRST METRIC"
  val SECOND_METRIC = "SECOND METRIC"
  val NUM_SAMPLES = 10
  val EXPECTED_SECTIONS = 2

}
