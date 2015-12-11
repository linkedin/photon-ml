package com.linkedin.photon.ml.diagnostics.independence

import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.diagnostics.ModelDiagnostic
import com.linkedin.photon.ml.stat.BasicStatisticalSummary
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import org.apache.spark.rdd.RDD

/**
 * Perform several tests of independence to see if prediction errors and predictions are independent.
 */
class PredictionErrorIndependenceDiagnostic extends ModelDiagnostic[GeneralizedLinearModel, PredictionErrorIndependenceReport] {
  import PredictionErrorIndependenceDiagnostic._

  override def diagnose(model: GeneralizedLinearModel, data: RDD[LabeledPoint], summary: Option[BasicStatisticalSummary]): PredictionErrorIndependenceReport = {
    val broadcastModel = data.sparkContext.broadcast(model)
    val predictionError = data.map( x => {
      val prediction = broadcastModel.value.computeMeanFunctionWithOffset(x.features, x.offset)
      val error = x.label - prediction
      (prediction, error)
    })

    val sample = predictionError.takeSample(false, MAXIMUM_SAMPLE_SIZE)
    val predictionSamples = sample.map(_._1)
    val errorSamples = sample.map(_._2)
    val kendallTau = KENDALL_TAU_ANALYSIS.analyze(sample)
    new PredictionErrorIndependenceReport(errorSamples, predictionSamples, kendallTau)
  }
}

object PredictionErrorIndependenceDiagnostic {
  val MAXIMUM_SAMPLE_SIZE = 5000
  val KENDALL_TAU_ANALYSIS = new KendallTauAnalysis
}
