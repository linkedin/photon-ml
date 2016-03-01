package com.linkedin.photon.ml.data

import org.apache.spark.storage.StorageLevel

import com.linkedin.photon.ml.RDDLike
import com.linkedin.photon.ml.projector.{ProjectorType, RandomEffectProjector}


/**
 * @author xazhang
 */
class RandomEffectDataSetInProjectedSpace(
    val randomEffectDataSetInProjectedSpace: RandomEffectDataSet,
    val randomEffectProjector: RandomEffectProjector)
    extends RandomEffectDataSet(
      randomEffectDataSetInProjectedSpace.activeData,
      randomEffectDataSetInProjectedSpace.globalIdToIndividualIds,
      randomEffectDataSetInProjectedSpace.passiveDataOption,
      randomEffectDataSetInProjectedSpace.passiveDataIndividualIdsOption,
      randomEffectDataSetInProjectedSpace.randomEffectId,
      randomEffectDataSetInProjectedSpace.featureShardId) {

  override def setName(name: String): this.type = {
    super.setName(name)
    randomEffectProjector match {
      case rddLike: RDDLike => rddLike.setName(s"$name: projector ${randomEffectProjector.getClass}")
      case _ =>
    }
    this
  }

  override def persistRDD(storageLevel: StorageLevel): this.type = {
    super.persistRDD(storageLevel)
    randomEffectProjector match {
      case rddLike: RDDLike => rddLike.persistRDD(storageLevel)
      case _ =>
    }
    this
  }

  override def unpersistRDD(): this.type = {
    randomEffectProjector match {
      case rddLike: RDDLike => rddLike.unpersistRDD()
      case _ =>
    }
    super.unpersistRDD()
    this
  }

  override def materialize(): this.type = {
    super.materialize()
    randomEffectProjector match {
      case rddLike: RDDLike => rddLike.materialize()
      case _ =>
    }
    this
  }
}

object RandomEffectDataSetInProjectedSpace {

  def buildWithProjectorType(
      randomEffectDataSet: RandomEffectDataSet,
      projectorType: ProjectorType): RandomEffectDataSetInProjectedSpace = {

    val randomEffectProjector = RandomEffectProjector.buildRandomEffectProjector(randomEffectDataSet, projectorType)
    val projectedRandomEffectDataSet = randomEffectProjector.projectRandomEffectDataSet(randomEffectDataSet)
    new RandomEffectDataSetInProjectedSpace(projectedRandomEffectDataSet, randomEffectProjector)
  }
}
