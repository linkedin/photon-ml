package com.linkedin.photon.ml.hyperparameter

import scala.math.{floor, max, min}

import breeze.linalg.{DenseMatrix, DenseVector}
import org.apache.commons.math3.random.SobolSequenceGenerator

import com.linkedin.photon.ml.hyperparameter.estimators.GaussianProcessEstimator
import com.linkedin.photon.ml.hyperparameter.estimators.kernels.Matern52

/**
  * An object to shrink search range given prior data.
  */
object ShrinkSearchRange {

  /**
    * Compute the lower bound and upper bound of the new prior config
    *
    * @param hyperParams Configurations of hyper-parameters
    * @param priorJsonString JSON string containing prior observations
    * @param priorDefault Default values for missing hyper-parameters
    * @param radius The radius of search range after transformation to [0, 1]
    * @return A tuple of lower bounds and upper bounds for each hyperparameter. Lower bounds and upper bounds are
    *         discrete for discrete parameters. Lower bounds and upper bounds are dense vectors.
    */
  def getBounds(
      hyperParams: HyperparameterConfig,
      priorJsonString: String,
      priorDefault: Map[String, String],
      radius: Double,
      candidatePoolSize: Int = 1000,
      seed: Long = System.currentTimeMillis): (DenseVector[Double], DenseVector[Double]) = {

    val hyperparameterList = hyperParams.names
    val ranges = hyperParams.ranges
    val discreteParams = hyperParams.discreteParams
    val numParams = ranges.length

    // Get a [[Seq]] of (vectorized hyper-parameters, evaluationValue) tuples
    val hyperparameterPairs = HyperparameterSerialization.priorFromJson(priorJsonString, priorDefault, hyperparameterList)

    // Rescale the hyperparameters to [0,1]
    val hyperparameterRescaled = VectorRescaling.rescalePriors(hyperparameterPairs, hyperParams)

    // Combine hyperparameters as a dense matrix and evaluation value as a dense vector
    val (overallPoints, overallEvals) = hyperparameterRescaled.map(x => (x._1.asDenseMatrix, DenseVector(x._2))).reduce(
      (a, b) => (DenseMatrix.vertcat(a._1, b._1), DenseVector.vertcat(a._2, b._2))
    )

    // Fit Gaussian process regression model
    val estimator = new GaussianProcessEstimator(kernel = new Matern52)
    val model = estimator.fit(overallPoints, overallEvals)

    // Sobol generator
    val paramDistributions = {
      val sobol = new SobolSequenceGenerator(numParams)
      sobol.skipTo((seed % (Int.MaxValue.toLong + 1)).toInt)
      sobol
    }

    // Draw candidates from a Sobol generator
    val candidates = (1 until candidatePoolSize).foldLeft(DenseMatrix(paramDistributions.nextVector)) { case (acc, _) =>
      DenseMatrix.vertcat(acc, DenseMatrix(paramDistributions.nextVector))
    }

    // Select the best candidate
    val predictions = model.predict(candidates)
    val bestCandidate = selectBestCandidate(candidates, predictions._1)

    // compute lower bound and upper bound
    val upperBound = VectorRescaling.scaleBackward(
      discretizeCandidate(bestCandidate.map(x => x + radius), discreteParams),
      ranges,
      discreteParams.keySet)
    val lowerBound = VectorRescaling.scaleBackward(
      discretizeCandidate(bestCandidate.map(x => x - radius), discreteParams),
      ranges,
      discreteParams.keySet)

    (0 until numParams).foreach{
      index =>
        upperBound(index) = min(upperBound(index), ranges(index).end)
        lowerBound(index) = max(lowerBound(index), ranges(index).start)
    }

    (lowerBound, upperBound)
  }

  /**
    * Selects the best candidate according to the predicted values, where "best" is defined as the largest
    *
    * @param candidates matrix of candidates
    * @param predictions predicted values for each candidate
    * @return the candidate with the best value
    */
  private def selectBestCandidate(
      candidates: DenseMatrix[Double],
      predictions: DenseVector[Double]): DenseVector[Double] = {

    val init = (candidates(0,::).t, predictions(0))

    val (selectedCandidate, _) = (1 until candidates.rows).foldLeft(init) {
      case ((bestCandidate, bestPrediction), i) =>
        if (predictions(i) > bestPrediction) {
          (candidates(i,::).t, predictions(i))
        } else {
          (bestCandidate, bestPrediction)
        }
    }

    selectedCandidate
  }

  /**
    * Discretize candidates with specified indices.
    *
    * @param candidate candidate with values in [0, 1]
    * @param discreteParams Map that specifies the indices of discrete parameters and their numbers of discrete values
    * @return candidate with the specified discrete values
    */
  private def discretizeCandidate(
      candidate: DenseVector[Double],
      discreteParams: Map[Int, Int]): DenseVector[Double] = {

    val candidateWithDiscrete = candidate.copy

    discreteParams.foreach { case (index, numDiscreteValues) =>
      candidateWithDiscrete(index) = floor(candidate(index) * numDiscreteValues) / numDiscreteValues
    }

    candidateWithDiscrete
  }
}
