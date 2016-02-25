package com.linkedin.photon.ml.algorithm


import breeze.linalg.Matrix
import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.contants.{MathConst, StorageLevel}
import com.linkedin.photon.ml.data._
import com.linkedin.photon.ml.function.EnhancedTwiceDiffFunction
import com.linkedin.photon.ml.model._
import com.linkedin.photon.ml.optimization._
import com.linkedin.photon.ml.projector.{ProjectionMatrixBroadcast, ProjectionMatrix}
import com.linkedin.photon.ml.util.VectorUtils


/**
 * @author xazhang
 */
class FactoredRandomEffectCoordinate[F <: EnhancedTwiceDiffFunction[LabeledPoint]](
    randomEffectDataSet: RandomEffectDataSet,
    factoredRandomEffectOptimizationProblem: FactoredRandomEffectOptimizationProblem[F])
    extends Coordinate[RandomEffectDataSet, FactoredRandomEffectCoordinate[F]](randomEffectDataSet) {

  override def initializeModel(seed: Long): Model = {
    val latentSpaceDimension = factoredRandomEffectOptimizationProblem.latentSpaceDimension
    FactoredRandomEffectCoordinate.initializeModel(randomEffectDataSet, latentSpaceDimension, seed)
  }

  override protected def updateModel(model: Model): (Model, OptimizationTracker) = {
    model match {
      case factoredRandomEffectModel: FactoredRandomEffectModel =>

        val numIterations = factoredRandomEffectOptimizationProblem.numIterations

        var updatedCoefficientsRDD = factoredRandomEffectModel.coefficientsRDDInProjectedSpace
        var updatedProjectionMatrixBroadcast = factoredRandomEffectModel.projectionMatrixBroadcast
        var latentFactorOptimizationProblem = factoredRandomEffectOptimizationProblem.latentFactorOptimizationProblem
        val factoredRandomEffectOptimizationTracker =
          new Array[(RandomEffectOptimizationTracker, FixedEffectOptimizationTracker)](numIterations)
        val randomEffectOptimizationProblem = factoredRandomEffectOptimizationProblem.randomEffectOptimizationProblem
        val sparkContext = randomEffectDataSet.sparkContext

        for (iteration <- 0 until numIterations) {
          // First update the coefficients
          val randomEffectDataSetInProjectedSpace =
            updatedProjectionMatrixBroadcast.projectRandomEffectDataSet(randomEffectDataSet)
          val randomEffectModel = factoredRandomEffectModel
              .updateRandomEffectModelInProjectedSpace(updatedCoefficientsRDD)
              .toRandomEffectModel

          val (updatedRandomEffectModel, randomEffectOptimizationTracker) =
            RandomEffectCoordinate.updateModel(randomEffectDataSetInProjectedSpace, randomEffectOptimizationProblem,
              randomEffectModel)
          updatedRandomEffectModel
              .coefficientsRDD
              .setName(s"Updated random effect model in iteration $iteration")
              .persist(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)
          randomEffectOptimizationTracker
              .setName(s"Random effect optimization tracker in iteration $iteration")
              .persistRDD(StorageLevel.INFREQUENT_REUSE_RDD_STORAGE_LEVEL)
          updatedRandomEffectModel.materialize()
          updatedCoefficientsRDD.unpersist()
          updatedCoefficientsRDD = updatedRandomEffectModel.coefficientsRDD

          // Then update the latent projection matrix
          val latentProjectionMatrix = updatedProjectionMatrixBroadcast.projectionMatrix
          val (updatedLatentProjectionMatrix, updatedOptimizationProblem) = FactoredRandomEffectCoordinate
              .updateLatentProjectionMatrix(randomEffectDataSet, updatedRandomEffectModel, latentProjectionMatrix,
                latentFactorOptimizationProblem)
          updatedProjectionMatrixBroadcast.unpersistBroadcast()
          val updatedLatentProjectionMatrixBroadcast = sparkContext.broadcast(updatedLatentProjectionMatrix)
          updatedProjectionMatrixBroadcast = new ProjectionMatrixBroadcast(updatedLatentProjectionMatrixBroadcast)
          //Note that the optimizationProblem will memorize the current state of optimization,
          //and the next round of updating latent factors will share the same convergence criteria as previous one.
          latentFactorOptimizationProblem = updatedOptimizationProblem

          val latentProjectionMatrixOptimizationStateTracker =
            new FixedEffectOptimizationTracker(latentFactorOptimizationProblem.optimizer.getStateTracker.get)
          factoredRandomEffectOptimizationTracker(iteration) = (randomEffectOptimizationTracker,
              latentProjectionMatrixOptimizationStateTracker)
        }
        // Return the updated model
        val updatedFactoredRandomEffectModel = factoredRandomEffectModel
            .updateFactoredRandomEffectModel(updatedCoefficientsRDD, updatedProjectionMatrixBroadcast)

        (updatedFactoredRandomEffectModel,
            new FactoredRandomEffectOptimizationTracker(factoredRandomEffectOptimizationTracker))
      case _ =>
        throw new UnsupportedOperationException(s"Updating model of type ${model.getClass} " +
            s"in ${this.getClass} is not supported!")
    }
  }

  override def score(model: Model): KeyValueScore = {
    model match {
      case factoredRandomEffectModel: FactoredRandomEffectModel =>
        val projectionMatrixBroadcast = factoredRandomEffectModel.projectionMatrixBroadcast
        val randomEffectModel = factoredRandomEffectModel.toRandomEffectModel
        val randomEffectDataSetInProjectedSpace =
          projectionMatrixBroadcast.projectRandomEffectDataSet(randomEffectDataSet)
        RandomEffectCoordinate.score(randomEffectDataSetInProjectedSpace, randomEffectModel)
      case _ =>
        throw new UnsupportedOperationException(s"Updating scores with model of type ${model.getClass} " +
            s"in ${this.getClass} is not supported!")
    }
  }

  override def computeRegularizationTermValue(model: Model): Double = {
    model match {
      case factoredRandomEffectModel: FactoredRandomEffectModel =>
        val coefficientsRDD = factoredRandomEffectModel.coefficientsRDDInProjectedSpace
        val projectionMatrix = factoredRandomEffectModel.projectionMatrixBroadcast.projectionMatrix
        factoredRandomEffectOptimizationProblem.getRegularizationTermValue(coefficientsRDD, projectionMatrix)
      case _ =>
        throw new UnsupportedOperationException(s"Compute the regularization term value with model of " +
            s"type ${model.getClass} in ${this.getClass} is not supported!")
    }
  }

  override protected def updateCoordinateWithDataSet(updatedRandomEffectDataSet: RandomEffectDataSet)
  : FactoredRandomEffectCoordinate[F] = {

    new FactoredRandomEffectCoordinate(updatedRandomEffectDataSet, factoredRandomEffectOptimizationProblem)
  }
}


object FactoredRandomEffectCoordinate {

  private def initializeModel(
      randomEffectDataSet: RandomEffectDataSet,
      latentSpaceDimension: Int,
      seed: Long): FactoredRandomEffectModel = {

    val randomEffectModel = randomEffectDataSet.activeData.mapValues(localDataSet =>
      Coefficients.initializeZeroCoefficients(latentSpaceDimension)
    )
    val numCols = latentSpaceDimension
    val latentProjectionMatrix = ProjectionMatrixBroadcast.buildRandomProjectionBroadcastProjector(randomEffectDataSet,
      numCols, isKeepingInterceptTerm = false, seed)
    val randomEffectId = randomEffectDataSet.randomEffectId
    val featureShardId = randomEffectDataSet.featureShardId
    new FactoredRandomEffectModel(randomEffectModel, latentProjectionMatrix, randomEffectId, featureShardId)
  }

  private def updateLatentProjectionMatrix[F <: EnhancedTwiceDiffFunction[LabeledPoint]](
      randomEffectDataSet: RandomEffectDataSet,
      randomEffectModel: RandomEffectModel,
      projectionMatrix: ProjectionMatrix,
      optimizationProblem: OptimizationProblem[F]): (ProjectionMatrix, OptimizationProblem[F]) = {

    val localDataSetRDD = randomEffectDataSet.activeData
    val coefficientsRDD = randomEffectModel.coefficientsRDD
    val latentProjectionMatrix = projectionMatrix.matrix
    val globalIdPartitioner = randomEffectDataSet.globalIdPartitioner
    val sampler = optimizationProblem.sampler
    val generatedTrainingData = crossProductFeaturesAndCoefficients(localDataSetRDD, coefficientsRDD,
      threshold = MathConst.LOW_PRECISION_TOLERANCE_THRESHOLD)
    val downSampledTrainingData = sampler.downSample(generatedTrainingData)
        .partitionBy(globalIdPartitioner)
        .setName("Generated training data for latent projection matrix")
        .persist(StorageLevel.FREQUENT_REUSE_RDD_STORAGE_LEVEL)
    val fixedEffectDataSet = new FixedEffectDataSet(downSampledTrainingData, featureShardId = "")

    val flattenedLatentProjectionMatrix = latentProjectionMatrix.flatten()
    val latentProjectionMatrixAsCoefficients = Coefficients(flattenedLatentProjectionMatrix, variancesOption = None)
    val (updatedCoefficients, _) = optimizationProblem.updatedCoefficientsMeans(downSampledTrainingData.values,
      latentProjectionMatrixAsCoefficients)
    downSampledTrainingData.unpersist()
    val numRows = latentProjectionMatrix.rows
    val numCols = latentProjectionMatrix.cols
    val updatedLatentProjectionMatrix = Matrix.create(numRows, numCols, updatedCoefficients.means.toArray)
    (new ProjectionMatrix(updatedLatentProjectionMatrix), optimizationProblem)
  }

  private def crossProductFeaturesAndCoefficients(
      localDataSetRDD: RDD[(String, LocalDataSet)],
      coefficientsRDD: RDD[(String, Coefficients)],
      threshold: Double): RDD[(Long, LabeledPoint)] = {

    localDataSetRDD.join(coefficientsRDD).flatMap { case (_, (localDataSet, coefficients)) =>
      localDataSet.dataPoints.map { case (globalId, labeledPoint) =>
        val generatedFeatures = VectorUtils.kroneckerProduct(labeledPoint.features, coefficients.means, threshold)
        val generatedLabeledPoint =
          new LabeledPoint(labeledPoint.label, generatedFeatures, labeledPoint.offset, labeledPoint.weight)
        (globalId, generatedLabeledPoint)
      }
    }
  }
}
