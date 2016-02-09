package com.linkedin.photon.ml.model

import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import com.linkedin.photon.ml.projector.RandomEffectProjector


/**
 * @author xazhang
 */
class RandomEffectModelInProjectedSpace(
    val coefficientsRDDInProjectedSpace: RDD[(String, Coefficients)],
    val randomEffectProjector: RandomEffectProjector,
    override val randomEffectId: String,
    override val featureShardId: String)
    extends RandomEffectModel(randomEffectProjector.projectCoefficientsRDD(coefficientsRDDInProjectedSpace),
      randomEffectId, featureShardId) {

  override def persistRDD(storageLevel: StorageLevel): this.type = {
    if (!coefficientsRDDInProjectedSpace.getStorageLevel.isValid) coefficientsRDDInProjectedSpace.persist(storageLevel)
    this
  }

  override def unpersistRDD(): this.type = {
    if (coefficientsRDDInProjectedSpace.getStorageLevel.isValid) coefficientsRDDInProjectedSpace.unpersist()
    this
  }

  override def setName(name: String): this.type = {
    coefficientsRDDInProjectedSpace.setName(name)
    this
  }

  override def materialize(): this.type = {
    coefficientsRDDInProjectedSpace.count()
    this
  }

  override def toSummaryString: String = {
    val stringBuilder = new StringBuilder(s"Random effect model with projector with randomEffectId $randomEffectId, " +
        s"featureShardId $featureShardId summary:")
    stringBuilder.append("\ncoefficientsRDDInProjectedSpace:")
    stringBuilder.append(s"\nLength: ${coefficientsRDDInProjectedSpace.values.map(_.means.length).stats()}")
    stringBuilder.append(s"\nMean: ${coefficientsRDDInProjectedSpace.map(_._2.meansL2Norm).stats()}")
    if (coefficientsRDDInProjectedSpace.first()._2.variancesOption.isDefined) {
      stringBuilder.append(s"\nVar: ${coefficientsRDDInProjectedSpace.map(_._2.variancesL2NormOption.get).stats()}")
    }
    stringBuilder.toString()
  }

  protected[ml] def toRandomEffectModel: RandomEffectModel = {
    new RandomEffectModel(coefficientsRDDInProjectedSpace, randomEffectId, featureShardId)
  }

  def updateRandomEffectModelInProjectedSpace(updatedCoefficientsRDDInProjectedSpace: RDD[(String, Coefficients)])
  : RandomEffectModelInProjectedSpace = {

    new RandomEffectModelInProjectedSpace(updatedCoefficientsRDDInProjectedSpace, randomEffectProjector, randomEffectId,
      featureShardId)
  }
}
