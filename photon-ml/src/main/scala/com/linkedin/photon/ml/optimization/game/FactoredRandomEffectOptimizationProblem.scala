package com.linkedin.photon.ml.optimization.game

import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import com.linkedin.photon.ml.RDDLike
import com.linkedin.photon.ml.data.{RandomEffectDataSet, LabeledPoint}
import com.linkedin.photon.ml.function.TwiceDiffFunction
import com.linkedin.photon.ml.model.Coefficients
import com.linkedin.photon.ml.projector.ProjectionMatrix
import com.linkedin.photon.ml.supervised.TaskType._

/**
 * An optimization problem for factored random effect datasets
 *
 * @param randomEffectOptimizationProblem the random effect optimization problem
 * @param latentFactorOptimizationProblem the latent factor optimization problem
 * @param numIterations number of iterations
 * @param latentSpaceDimension dimensionality of latent space
 * @author xazhang
 */
class FactoredRandomEffectOptimizationProblem[F <: TwiceDiffFunction[LabeledPoint]](
    val randomEffectOptimizationProblem: RandomEffectOptimizationProblem[F],
    val latentFactorOptimizationProblem: OptimizationProblem[F],
    val numIterations: Int,
    val latentSpaceDimension: Int)
  extends RDDLike {

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

  /**
   * Compute the regularization term value
   *
   * @param model the model
   * @return regularization term value
   */
  def getRegularizationTermValue(
      coefficientsRDD: RDD[(String, Coefficients)],
      projectionMatrix: ProjectionMatrix): Double = {

    val projectionMatrixAsCoefficients = new Coefficients(projectionMatrix.matrix.flatten(), variancesOption = None)
    randomEffectOptimizationProblem.getRegularizationTermValue(coefficientsRDD) +
        latentFactorOptimizationProblem.getRegularizationTermValue(projectionMatrixAsCoefficients)
  }
}

object FactoredRandomEffectOptimizationProblem {

  /**
   * Builds a factored random effect optimization problem
   *
   * @param taskType the task type
   * @param randomEffectOptimizationConfiguration random effect configuration
   * @param latentFactorOptimizationConfiguration latent factor configuration
   * @param mfOptimizationConfiguration MF configuration
   * @param randomEffectDataSet the dataset
   * @return the new optimization problem
   */
  def buildFactoredRandomEffectOptimizationProblem(
      taskType: TaskType,
      randomEffectOptimizationConfiguration: GLMOptimizationConfiguration,
      latentFactorOptimizationConfiguration: GLMOptimizationConfiguration,
      mfOptimizationConfiguration: MFOptimizationConfiguration,
      randomEffectDataSet: RandomEffectDataSet)
  : FactoredRandomEffectOptimizationProblem[TwiceDiffFunction[LabeledPoint]] = {

    val randomEffectOptimizationProblem = RandomEffectOptimizationProblem.buildRandomEffectOptimizationProblem(taskType,
      randomEffectOptimizationConfiguration, randomEffectDataSet)
    val latentFactorOptimizationProblem = OptimizationProblem.buildOptimizationProblem(taskType,
      latentFactorOptimizationConfiguration)
    val MFOptimizationConfiguration(numInnerIterations, latentSpaceDimension) = mfOptimizationConfiguration
    new FactoredRandomEffectOptimizationProblem(randomEffectOptimizationProblem, latentFactorOptimizationProblem,
      numInnerIterations, latentSpaceDimension)
  }
}
