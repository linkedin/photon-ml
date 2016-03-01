package com.linkedin.photon.ml.optimization.game

import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import com.linkedin.photon.ml.RDDLike
import com.linkedin.photon.ml.data.{RandomEffectDataSet, LabeledPoint}
import com.linkedin.photon.ml.function.TwiceDiffFunction
import com.linkedin.photon.ml.model.Coefficients
import com.linkedin.photon.ml.supervised.TaskType.TaskType


/**
 * @author xazhang
 *         Why sharding the optimizers? Because we may want to preserve the optimization state of each sharded
 *         optimization problem
 *         Why sharding the objective functions? Because the regularization weight for each sharded optimization
 *         problem may be different, which leads to different objective functions.
 */
class RandomEffectOptimizationProblem[F <: TwiceDiffFunction[LabeledPoint]](
    val optimizationProblems: RDD[(String, OptimizationProblem[F])]) extends RDDLike {

  def sparkContext = optimizationProblems.sparkContext

  def setName(name: String): this.type = {
    optimizationProblems.setName(s"$name: Optimization problems")
    this
  }

  def persistRDD(storageLevel: StorageLevel): this.type = {
    if (!optimizationProblems.getStorageLevel.isValid) {
      optimizationProblems.persist(storageLevel)
    }
    this
  }

  def unpersistRDD(): this.type = {
    if (optimizationProblems.getStorageLevel.isValid) {
      optimizationProblems.unpersist()
    }
    this
  }

  def materialize(): this.type = {
    optimizationProblems.count()
    this
  }

  def getRegularizationTermValue(coefficientsRDD: RDD[(String, Coefficients)]): Double = {
    optimizationProblems
      .join(coefficientsRDD)
      .map { case (_, (optimizationProblem, coefficients)) =>
      optimizationProblem.getRegularizationTermValue(coefficients)
    }
      .reduce(_ + _)
  }
}

object RandomEffectOptimizationProblem {
  def buildRandomEffectOptimizationProblem(
      taskType: TaskType,
      configuration: GLMOptimizationConfiguration,
      randomEffectDataSet: RandomEffectDataSet)
  : RandomEffectOptimizationProblem[TwiceDiffFunction[LabeledPoint]] = {

    val optimizationProblems = randomEffectDataSet.activeData.mapValues(_ =>
      OptimizationProblem.buildOptimizationProblem(taskType, configuration)
    )
    new RandomEffectOptimizationProblem(optimizationProblems)
  }
}
