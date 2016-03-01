package com.linkedin.photon.ml.model


import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.rdd.RDD._
import org.apache.spark.storage.StorageLevel
import com.linkedin.photon.ml.RDDLike
import com.linkedin.photon.ml.data.{KeyValueScore, GameData}


/**
 * @author xazhang
 */
class RandomEffectModel(
    val coefficientsRDD: RDD[(String, Coefficients)],
    val randomEffectId: String,
    val featureShardId: String) extends Model with RDDLike {

  override def sparkContext: SparkContext = coefficientsRDD.sparkContext

  override def persistRDD(storageLevel: StorageLevel): this.type = {
    if (!coefficientsRDD.getStorageLevel.isValid) coefficientsRDD.persist(storageLevel)
    this
  }

  override def unpersistRDD(): this.type = {
    if (coefficientsRDD.getStorageLevel.isValid) coefficientsRDD.unpersist()
    this
  }

  override def setName(name: String): this.type = {
    coefficientsRDD.setName(name)
    this
  }

  override def materialize(): this.type = {
    coefficientsRDD.count()
    this
  }

  override def score(dataPoints: RDD[(Long, GameData)]): KeyValueScore = {
    RandomEffectModel.score(dataPoints, coefficientsRDD, randomEffectId, featureShardId)
  }

  override def toSummaryString: String = {
    val stringBuilder = new StringBuilder(s"Random effect model of randomEffectId $randomEffectId, " +
        s"featureShardId $featureShardId summary:")
    stringBuilder.append(s"\nLength: ${coefficientsRDD.values.map(_.means.length).stats()}")
    stringBuilder.append(s"\nMean: ${coefficientsRDD.map(_._2.meansL2Norm).stats()}")
    if (coefficientsRDD.first()._2.variancesOption.isDefined) {
      stringBuilder.append(s"\nvariance: ${coefficientsRDD.map(_._2.variancesL2NormOption.get).stats()}")
    }
    stringBuilder.toString()
  }

  def updateRandomEffectModel(updatedCoefficientsRDD: RDD[(String, Coefficients)]): RandomEffectModel = {
    new RandomEffectModel(updatedCoefficientsRDD, randomEffectId, featureShardId)
  }
}

object RandomEffectModel {

  protected def score(
      dataPoints: RDD[(Long, GameData)],
      coefficientsRDD: RDD[(String, Coefficients)],
      randomEffectId: String,
      featureShardId: String): KeyValueScore = {

    val scores = dataPoints.map { case (globalId, gameData) =>
      val individualId = gameData.randomEffectIdToIndividualIdMap(randomEffectId)
      val features = gameData.featureShardContainer(featureShardId)
      (individualId, (globalId, features))
    }.cogroup(coefficientsRDD).flatMap { case (individualId, (globalIdAndFeaturesIterable, coefficientsIterable)) =>
      assert(coefficientsIterable.size <= 1, s"More than one coefficients (${coefficientsIterable.size}) " +
          s"found for individual Id $individualId of random effect Id $randomEffectId")
      if (coefficientsIterable.isEmpty) {
        globalIdAndFeaturesIterable.map { case (globalId, _) => (globalId, 0.0) }
      } else {
        val coefficients = coefficientsIterable.head
        globalIdAndFeaturesIterable.map { case (globalId, features) => (globalId, coefficients.computeScore(features)) }
      }
    }
    new KeyValueScore(scores)
  }
}
