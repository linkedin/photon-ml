package com.linkedin.photon.ml.optimization.game

import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import com.linkedin.photon.ml.RDDLike
import com.linkedin.photon.ml.data.{RandomEffectDataSet, LabeledPoint}
import com.linkedin.photon.ml.function.EnhancedTwiceDiffFunction
import com.linkedin.photon.ml.model.Coefficients
import com.linkedin.photon.ml.projector.ProjectionMatrix
import com.linkedin.photon.ml.supervised.TaskType._


/**
 * @author xazhang
 */
class FactoredRandomEffectOptimizationProblem[F <: EnhancedTwiceDiffFunction[LabeledPoint]](
    val randomEffectOptimizationProblem: RandomEffectOptimizationProblem[F],
    val latentFactorOptimizationProblem: OptimizationProblem[F],
    val numIterations: Int,
    val latentSpaceDimension: Int) extends RDDLike {

  override def sparkContext = randomEffectOptimizationProblem.sparkContext

  override def setName(name: String): this.type = {
    randomEffectOptimizationProblem.setName(name)
    this
  }

  override def persistRDD(storageLevel: StorageLevel): this.type = {
    randomEffectOptimizationProblem.persistRDD(storageLevel)
    this
  }

  override def unpersistRDD(): this.type = {
    randomEffectOptimizationProblem.unpersistRDD()
    this
  }

  override def materialize(): this.type = {
    randomEffectOptimizationProblem.materialize()
    this
  }

  def getRegularizationTermValue(
      coefficientsRDD: RDD[(String, Coefficients)],
      projectionMatrix: ProjectionMatrix): Double = {

    val projectionMatrixAsCoefficients = new Coefficients(projectionMatrix.matrix.flatten(), variancesOption = None)
    randomEffectOptimizationProblem.getRegularizationTermValue(coefficientsRDD) +
        latentFactorOptimizationProblem.getRegularizationTermValue(projectionMatrixAsCoefficients)
  }
}

object FactoredRandomEffectOptimizationProblem {
  def buildFactoredRandomEffectOptimizationProblem(
      taskType: TaskType,
      randomEffectOptimizationConfiguration: GLMOptimizationConfiguration,
      latentFactorOptimizationConfiguration: GLMOptimizationConfiguration,
      mfOptimizationConfiguration: MFOptimizationConfiguration,
      randomEffectDataSet: RandomEffectDataSet)
  : FactoredRandomEffectOptimizationProblem[EnhancedTwiceDiffFunction[LabeledPoint]] = {

    val randomEffectOptimizationProblem = RandomEffectOptimizationProblem.buildRandomEffectOptimizationProblem(taskType,
      randomEffectOptimizationConfiguration, randomEffectDataSet)
    val latentFactorOptimizationProblem = OptimizationProblem.buildOptimizationProblem(taskType,
      latentFactorOptimizationConfiguration)
    val MFOptimizationConfiguration(numInnerIterations, latentSpaceDimension) = mfOptimizationConfiguration
    new FactoredRandomEffectOptimizationProblem(randomEffectOptimizationProblem, latentFactorOptimizationProblem,
      numInnerIterations, latentSpaceDimension)
  }
}
