package com.linkedin.photon.ml.diagnostics.bootstrap

import com.linkedin.photon.ml.Evaluation
import com.linkedin.photon.ml.diagnostics.reporting.SectionPhysicalReport
import com.linkedin.photon.ml.supervised.model.CoefficientSummary
import org.testng.Assert._
import org.testng.annotations._

class BootstrapToPhysicalReportTransformerTest {
  @Test
  def checkAllEmpty(): Unit = {
    val metricDistributions = Map[String, (Double, Double, Double, Double, Double)]()
    val bootstrapModelMetrics = Map[String, Double]()
    val importantFeatureCoefficientDistributions = Map[(String, String), CoefficientSummary]()
    val zeroCrossingFeatures = Map[(String, String), (Int, Double, CoefficientSummary)]()

    val report = BootstrapReport(
      metricDistributions = metricDistributions,
      bootstrappedModelMetrics = bootstrapModelMetrics,
      importantFeatureCoefficientDistributions = importantFeatureCoefficientDistributions,
      zeroCrossingFeatures = zeroCrossingFeatures)

    val transformer = new BootstrapToPhysicalReportTransformer

    val transformed = transformer.transform(report)

    assertEquals(transformed.title, BootstrapToPhysicalReportTransformer.BOOTSTRAP_SECTION_TITLE)
    assertEquals(transformed.items.size, 4)
    transformed.items.foreach(x => {
      assertTrue(x.isInstanceOf[SectionPhysicalReport])
    })
  }

  @Test
  def checkHappyPath(): Unit = {
    val summary = new CoefficientSummary
    summary.accumulate(1.0)
    val metricDistributions = Map[String, (Double, Double, Double, Double, Double)](Evaluation.AREA_UNDER_PRECISION_RECALL -> (1.0, 1.0, 1.0, 1.0, 1.0))
    val bootstrapModelMetrics = Map[String, Double](Evaluation.AREA_UNDER_PRECISION_RECALL -> 1.0)
    val importantFeatureCoefficientDistributions = Map[(String, String), CoefficientSummary](("Name 1", "Term 1") -> summary)
    val zeroCrossingFeatures = Map[(String, String), (Int, Double, CoefficientSummary)]()

    val report = BootstrapReport(
      metricDistributions = metricDistributions,
      bootstrappedModelMetrics = bootstrapModelMetrics,
      importantFeatureCoefficientDistributions = importantFeatureCoefficientDistributions,
      zeroCrossingFeatures = zeroCrossingFeatures)

    val transformer = new BootstrapToPhysicalReportTransformer

    val transformed = transformer.transform(report)

    assertEquals(transformed.title, BootstrapToPhysicalReportTransformer.BOOTSTRAP_SECTION_TITLE)
    assertEquals(transformed.items.size, 4)
    transformed.items.foreach(x => {
      assertTrue(x.isInstanceOf[SectionPhysicalReport])
    })
  }
}
