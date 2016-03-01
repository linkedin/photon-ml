package com.linkedin.photon.ml.model


import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.BroadcastLike
import com.linkedin.photon.ml.projector.ProjectionMatrixBroadcast


/**
 * @author xazhang
 */
class FactoredRandomEffectModel(
    override val coefficientsRDDInProjectedSpace: RDD[(String, Coefficients)],
    val projectionMatrixBroadcast: ProjectionMatrixBroadcast,
    override val randomEffectId: String,
    override val featureShardId: String)
    extends RandomEffectModelInProjectedSpace(coefficientsRDDInProjectedSpace, projectionMatrixBroadcast,
      randomEffectId, featureShardId) with BroadcastLike {

  override def unpersistBroadcast(): this.type = {
    projectionMatrixBroadcast.unpersistBroadcast()
    this
  }

  override def toSummaryString: String = {
    val stringBuilder = new StringBuilder(super.toSummaryString)
    stringBuilder.append("\nprojectionMatrix:")
    stringBuilder.append(s"\n${projectionMatrixBroadcast.projectionMatrix.toSummaryString}")
    stringBuilder.toString()
  }

  def updateFactoredRandomEffectModel(
      updatedCoefficientsRDDInProjectedSpace: RDD[(String, Coefficients)],
      updatedProjectionMatrixBroadcast: ProjectionMatrixBroadcast): FactoredRandomEffectModel = {
    new FactoredRandomEffectModel(updatedCoefficientsRDDInProjectedSpace, updatedProjectionMatrixBroadcast,
      randomEffectId, featureShardId)
  }
}
