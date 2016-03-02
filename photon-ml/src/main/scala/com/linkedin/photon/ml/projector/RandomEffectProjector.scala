package com.linkedin.photon.ml.projector

import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.data.RandomEffectDataSet
import com.linkedin.photon.ml.model.Coefficients


/**
 * A trait that performs two types of projections:
 * <ul>
 * <li>
 *   Project the random effect data set from the original space to the projected space, usually at a pre-processing
 *   step before the model training phase.
 * </li>
 * <li>
 *   Project the coefficients from the projected space back to the original space, usually after training the model
 *   or before scoring a data set in the original space.
 * </li>
 * </ul>
 * @author xazhang
 */
trait RandomEffectProjector {
  /**
   * Project the sharded data set from the original space to the projected space
   * @param randomEffectDataSet The input sharded data set in the original space
   * @return The sharded data set in the projected space
   */
  def projectRandomEffectDataSet(randomEffectDataSet: RandomEffectDataSet): RandomEffectDataSet

  /**
   * Project a [[RDD]] of [[Coefficients]] from the projected space back to the original space
   * @param coefficientsRDD The input [[RDD]] of [[Coefficients]] in the projected space
   * @return The [[RDD]] of [[Coefficients]] in the original space
   */
  def projectCoefficientsRDD(coefficientsRDD: RDD[(String, Coefficients)]): RDD[(String, Coefficients)]
}

object RandomEffectProjector {

  /**
   * Builds a random effect projector instance
   *
   * @param randomEffectDataSet the dataset
   * @param projectorType
   * @return the projector
   */
  def buildRandomEffectProjector(
      randomEffectDataSet: RandomEffectDataSet,
      projectorType: ProjectorType): RandomEffectProjector = {

    projectorType match {
      case RandomProjection(projectedSpaceDimension) =>
        ProjectionMatrixBroadcast.buildRandomProjectionBroadcastProjector(
          randomEffectDataSet, projectedSpaceDimension, isKeepingInterceptTerm = true)

      case IndexMapProjection => IndexMapProjectorRDD.buildIndexMapProjector(randomEffectDataSet)
      case _ => throw new UnsupportedOperationException(s"Projector type $projectorType for random effect data set " +
          s"is not supported!")
    }
  }
}
