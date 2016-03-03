package com.linkedin.photon.ml.projector

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import com.linkedin.photon.ml.RDDLike
import com.linkedin.photon.ml.data.{RandomEffectDataSet, LabeledPoint}
import com.linkedin.photon.ml.model.Coefficients

/**
 * A class that holds the projectors for a sharded data set
 *
 * @param indexMapProjectorRDD The projectors
 */
class IndexMapProjectorRDD(indexMapProjectorRDD: RDD[(String, IndexMapProjector)])
  extends RandomEffectProjector with RDDLike {

  /**
   * Projects the random effect dataset into the new space
   *
   * @param randomEffectDataSet the dataset
   * @return the projected dataset
   */
  override def projectRandomEffectDataSet(randomEffectDataSet: RandomEffectDataSet): RandomEffectDataSet = {
    val activeData = randomEffectDataSet.activeData
    val passiveDataOption = randomEffectDataSet.passiveDataOption
    val passiveDataIndividualIdsOption = randomEffectDataSet.passiveDataIndividualIdsOption
    val projectedActiveData =
      activeData
        // Make sure the activeData retains its partitioner, especially when the partitioner of featureMaps is
        // not the same as that of activeData
        .join(indexMapProjectorRDD, activeData.partitioner.get)
        .mapValues { case (localDataSet, projector) => localDataSet.projectFeatures(projector) }

    val projectedPassiveData =
      if (passiveDataOption.isDefined) {
        val passiveDataIndividualIds = passiveDataIndividualIdsOption.get
        val projectorsForPassiveData = indexMapProjectorRDD.filter { case (individualId, _) =>
          passiveDataIndividualIds.value.contains(individualId)
        }.collectAsMap()

        //TODO: When and how to properly unpersist the broadcasted variables afterwards
        val projectorsForPassiveDataBroadcast = passiveDataOption.get.sparkContext.broadcast(projectorsForPassiveData)
        passiveDataOption.map(_.mapValues { case (shardId, LabeledPoint(response, features, offset, weight)) =>
          val projector = projectorsForPassiveDataBroadcast.value(shardId)
          val projectedFeatures = projector.projectFeatures(features)

          (shardId, LabeledPoint(response, projectedFeatures, offset, weight))
        })
      } else {
        None
      }

    randomEffectDataSet.update(projectedActiveData, projectedPassiveData)
  }

  /**
   * Project the coefficients RDD into the new space
   *
   * @param coefficientsRDD the coefficients
   * @return projected coefficients
   */
  override def projectCoefficientsRDD(coefficientsRDD: RDD[(String, Coefficients)]): RDD[(String, Coefficients)] = {
    coefficientsRDD.join(indexMapProjectorRDD)
        .mapValues { case (Coefficients(mean, varianceOption), projector) =>
      Coefficients(projector.projectCoefficients(mean), varianceOption.map(projector.projectCoefficients))
    }
  }

  override def sparkContext: SparkContext = {
    indexMapProjectorRDD.sparkContext
  }

  override def setName(name: String): this.type = {
    indexMapProjectorRDD.setName(name)
    this
  }

  override def persistRDD(storageLevel: StorageLevel): this.type = {
    if (!indexMapProjectorRDD.getStorageLevel.isValid) indexMapProjectorRDD.persist(storageLevel)
    this
  }

  override def unpersistRDD(): this.type = {
    if (indexMapProjectorRDD.getStorageLevel.isValid) indexMapProjectorRDD.unpersist()
    this
  }

  override def materialize(): this.type = {
    indexMapProjectorRDD.count()
    this
  }
}

object IndexMapProjectorRDD {

  /**
   * Generate index map based RDD projectors
   *
   * @param randomEffectDataSet The input random effect data set
   * @return The generated index map based RDD projectors
   */
  def buildIndexMapProjector(randomEffectDataSet: RandomEffectDataSet): IndexMapProjectorRDD = {
    val indexMapProjectors = randomEffectDataSet.activeData.mapValues(localDataSet =>
      IndexMapProjector.buildIndexMapProjector(localDataSet.dataPoints.map(_._2.features))
    )
    new IndexMapProjectorRDD(indexMapProjectors)
  }
}
