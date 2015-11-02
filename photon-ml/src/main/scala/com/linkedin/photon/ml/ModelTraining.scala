package com.linkedin.photon.ml

import com.linkedin.photon.ml.DataValidationType.DataValidationType
import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.normalization.NormalizationContext
import com.linkedin.photon.ml.optimization.OptimizerType.OptimizerType
import com.linkedin.photon.ml.optimization.{LBFGS, OptimizerType, RegularizationContext, TRON}
import com.linkedin.photon.ml.supervised.TaskType._
import com.linkedin.photon.ml.supervised.classification.LogisticRegressionAlgorithm
import com.linkedin.photon.ml.supervised.model.{GeneralizedLinearModel, ModelTracker}
import com.linkedin.photon.ml.supervised.regression.{LinearRegressionAlgorithm, PoissonRegressionAlgorithm}
import org.apache.spark.rdd.RDD


/**
 * Collection of functions for model training
 * @author xazhang
 * @author dpeng
 */
object ModelTraining {

  /**
   * Train a generalized linear model using the given training data set and the Photon-ML's parameter settings
   * @param trainingData The training data represented as a RDD of [[data.LabeledPoint]]
   * @param taskType Learning task type, e.g., LINEAR_REGRESSION or BINARY_CLASSIFICATION or POISSON_REGRESSION
   * @param optimizerType The type of optimizer that will be used to train the model
   * @param regularizationContext The type of regularization that will be used to train the model
   * @param regularizationWeights An array of regularization weights used to train the model
   * @param normalizationContext Normalization context for feature normalization
   * @param maxNumIter Maximum number of iterations to run
   * @param tolerance The optimizer's convergence tolerance, smaller value will lead to higher accuracy with the cost of more iterations
   * @param enableOptimizationStateTracker Whether to enable the optimization state tracker, which stores the per-iteration log information of the running optimizer
   * @return The trained models in the form of Map(key -> model), where key is the String typed corresponding regularization weight used to train the model
   */
  def trainGeneralizedLinearModel(trainingData: RDD[LabeledPoint],
                                  taskType: TaskType,
                                  optimizerType: OptimizerType,
                                  regularizationContext: RegularizationContext,
                                  regularizationWeights: List[Double],
                                  normalizationContext: NormalizationContext,
                                  maxNumIter: Int,
                                  tolerance: Double,
                                  enableOptimizationStateTracker: Boolean,
                                  dataValidationType: DataValidationType,
                                  constraintMap: Option[Map[Int, (Double, Double)]]): (List[(Double, _ <: GeneralizedLinearModel)], Option[List[(Double, ModelTracker)]]) = {
    /* Choose the optimizer */
    val optimizer = optimizerType match {
      case OptimizerType.LBFGS =>
        new LBFGS[LabeledPoint]
      case OptimizerType.TRON =>
        new TRON[LabeledPoint]
      case optType =>
        throw new IllegalArgumentException(s"Optimizer type unrecognized: $optType.");
    }
    optimizer.maxNumIterations = maxNumIter
    optimizer.tolerance = tolerance
    optimizer.constraintMap = constraintMap

    /* Choose the generalized linear algorithm */
    val algorithm = taskType match {
      case LINEAR_REGRESSION => new LinearRegressionAlgorithm
      case POISSON_REGRESSION => new PoissonRegressionAlgorithm
      case LOGISTIC_REGRESSION => new LogisticRegressionAlgorithm
      case _ => throw new IllegalArgumentException(s"unrecognized task type $taskType")
    }
    algorithm.isTrackingState = enableOptimizationStateTracker
    /* Sort the regularization weights from high to low, which would potentially speed up the overall convergence time */
    val sortedRegularizationWeights = regularizationWeights.sortWith(_ >= _)
    /* Model training with the chosen optimizer and algorithm */
    val models = algorithm.run(trainingData, optimizer, regularizationContext, sortedRegularizationWeights, normalizationContext, dataValidationType)
    val weightModelTuples = sortedRegularizationWeights.zip(models)

    val modelTrackersMapOption = algorithm.getStateTracker.map(modelTrackers => sortedRegularizationWeights.zip(modelTrackers))
    (weightModelTuples, modelTrackersMapOption)
  }
}
