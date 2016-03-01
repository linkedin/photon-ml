package com.linkedin.photon.ml.projector


import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.BroadcastLike
import com.linkedin.photon.ml.constants.MathConst
import com.linkedin.photon.ml.data.{RandomEffectDataSet, LabeledPoint}
import com.linkedin.photon.ml.model.Coefficients


/**
 * @author xazhang
 */
class ProjectionMatrixBroadcast(projectionMatrixBroadcast: Broadcast[ProjectionMatrix])
    extends RandomEffectProjector with BroadcastLike with Serializable {

  val projectionMatrix = projectionMatrixBroadcast.value

  override def projectRandomEffectDataSet(randomEffectDataSet: RandomEffectDataSet): RandomEffectDataSet = {
    val activeData = randomEffectDataSet.activeData
    val passiveDataOption = randomEffectDataSet.passiveDataOption
    val projectedActiveData = activeData.mapValues(_.projectFeatures(projectionMatrixBroadcast.value))
    val projectedPassiveData = if (passiveDataOption.isDefined) {
      passiveDataOption.map(_.mapValues { case (shardId, LabeledPoint(response, features, offset, weight)) =>
        val projectedFeatures = projectionMatrixBroadcast.value.projectFeatures(features)
        (shardId, LabeledPoint(response, projectedFeatures, offset, weight))
      })
    } else {
      None
    }
    randomEffectDataSet.update(projectedActiveData, projectedPassiveData)
  }

  override def projectCoefficientsRDD(coefficientsRDD: RDD[(String, Coefficients)]): RDD[(String, Coefficients)] = {
    coefficientsRDD.mapValues { case Coefficients(mean, varianceOption) =>
      Coefficients(projectionMatrixBroadcast.value.projectCoefficients(mean), varianceOption)
    }
  }

  override def unpersistBroadcast(): this.type = {
    projectionMatrixBroadcast.unpersist()
    this
  }
}

object ProjectionMatrixBroadcast {

  /**
   * Generate random projection based broadcast projector
   * @param randomEffectDataSet The input random effect data set
   * @param projectedSpaceDimension The dimension of the projected feature space
   * @param isKeepingInterceptTerm Whether to keep the intercept in the original feature space
   * @param seed The seed of random number generator
   * @return The generated random projection based broadcast projector
   */
  def buildRandomProjectionBroadcastProjector(
      randomEffectDataSet: RandomEffectDataSet,
      projectedSpaceDimension: Int,
      isKeepingInterceptTerm: Boolean,
      seed: Long = MathConst.RANDOM_SEED): ProjectionMatrixBroadcast = {

    val sparkContext = randomEffectDataSet.sparkContext
    val originalSpaceDimension = randomEffectDataSet.activeData.first()._2.numFeatures
    val randomProjectionMatrix = ProjectionMatrix.buildGaussianRandomProjectionMatrix(projectedSpaceDimension,
      originalSpaceDimension, isKeepingInterceptTerm, seed)
    val randomProjectionMatrixBroadcast = sparkContext.broadcast[ProjectionMatrix](randomProjectionMatrix)
    new ProjectionMatrixBroadcast(randomProjectionMatrixBroadcast)
  }
}
