package com.linkedin.photon.ml.diagnostics.reporting.reports.system

import java.util.Date

import com.linkedin.photon.ml.Params
import com.linkedin.photon.ml.diagnostics.reporting.{
  BulletedListPhysicalReport, SimpleTextPhysicalReport, SectionPhysicalReport, LogicalToPhysicalReportTransformer}

/**
 * Convert parameters into a presentable form.
 */
class ParametersToPhysicalReportTransformer
  extends LogicalToPhysicalReportTransformer[ParametersReport, SectionPhysicalReport] {

  import ParametersToPhysicalReportTransformer._

  def transform(param: ParametersReport): SectionPhysicalReport = {
    val contents = render(param.parameters)
    new SectionPhysicalReport(Seq(contents), PARAMETERS_SECTION_HEADER)
  }

  private def render(params: Params): BulletedListPhysicalReport = {
    val jobParameters = new BulletedListPhysicalReport(
      Seq(
        new SimpleTextPhysicalReport("Job parameters"),
        new BulletedListPhysicalReport(
          Seq(
            new SimpleTextPhysicalReport(s"Job name: ${params.jobName}"),
            new SimpleTextPhysicalReport(s"Run date: " + (new Date()).toString),
            new SimpleTextPhysicalReport(s"Kryo serialization: ${params.kryo}")
          )
        )
      )
    )

    val ioParameters = new BulletedListPhysicalReport(
      Seq(
        new SimpleTextPhysicalReport("I/O parameters"),
        new BulletedListPhysicalReport(
          Seq(
            new SimpleTextPhysicalReport(s"Training data directory: ${params.trainDir}"),
            new SimpleTextPhysicalReport(s"Output directory: ${params.outputDir}"),
            new SimpleTextPhysicalReport(s"Field name type: ${params.fieldsNameType}")
          )
            ++ params.validateDirOpt.toSeq.map(x => new SimpleTextPhysicalReport(s"Validation data directory: $x"))
            ++ params.summarizationOutputDirOpt.toSeq
                .map(x => new SimpleTextPhysicalReport(s"Summarization output directory: $x"))
        )
      )
    )

    val modelParameters = new BulletedListPhysicalReport(
      Seq(
        new SimpleTextPhysicalReport("Modeling parameters"),
        new BulletedListPhysicalReport(
          Seq(
            new SimpleTextPhysicalReport(s"Model type: ${params.taskType}"),
            new SimpleTextPhysicalReport(s"Add intercept: ${params.addIntercept}"),
            new SimpleTextPhysicalReport(s"Normalization type: ${params.normalizationType}"),
            new SimpleTextPhysicalReport(s"Regularization type: ${params.regularizationType}"),
            new SimpleTextPhysicalReport(s"Regularization weight: ${params.regularizationWeights}"),
            new SimpleTextPhysicalReport(s"Optimizer type: ${params.optimizerType}"),
            new SimpleTextPhysicalReport(s"Tolerance: ${params.tolerance}"),
            new SimpleTextPhysicalReport(s"Max iterations: ${params.maxNumIter}")
          )
            ++ params.constraintString.toSeq.map(x => new SimpleTextPhysicalReport(s"Constraints: $x"))
            ++ params.elasticNetAlpha.toSeq.map(x => new SimpleTextPhysicalReport(s"Alpha for elastic net: $x"))
        )
      )
    )

    new BulletedListPhysicalReport(Seq(jobParameters, ioParameters, modelParameters))
  }
}

object ParametersToPhysicalReportTransformer {
  val PARAMETERS_SECTION_HEADER = "Command-line options"
}
