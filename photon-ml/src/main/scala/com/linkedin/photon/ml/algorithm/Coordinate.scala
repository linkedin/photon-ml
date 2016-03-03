package com.linkedin.photon.ml.algorithm

import com.linkedin.photon.ml.data.{KeyValueScore, DataSet}
import com.linkedin.photon.ml.model.Model
import com.linkedin.photon.ml.optimization.game.OptimizationTracker


/**
 * The optimization problem coordinate for each effect model
 *
 * @param dataSet the training dataset
 * @author xazhang
 */
abstract class Coordinate[D <: DataSet[D], C <: Coordinate[D, C]](dataSet: D) {

  /**
   * Score the data set in the coordinate with the input model
   *
   * @param model the input model
   * @return the output scores
   */
  def score(model: Model): KeyValueScore

  /**
   * Initialize the model
   *
   * @param seed random seed
   */
  def initializeModel(seed: Long): Model

  def updateModel(model: Model, score: KeyValueScore): (Model, OptimizationTracker) = {
    val dataSetWithUpdatedOffsets = dataSet.addScoresToOffsets(score)
    updateCoordinateWithDataSet(dataSetWithUpdatedOffsets).updateModel(model)
  }

  protected def updateCoordinateWithDataSet(dataSet: D): C

  protected def updateModel(model: Model): (Model, OptimizationTracker)

  def computeRegularizationTermValue(model: Model): Double
}
