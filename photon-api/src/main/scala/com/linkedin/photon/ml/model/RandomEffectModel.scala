/*
 * Copyright 2017 LinkedIn Corp. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License"); you may
 * not use this file except in compliance with the License. You may obtain a
 * copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */
package com.linkedin.photon.ml.model

import org.apache.spark.ml.linalg.{Vector => SparkVector}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.col

import com.linkedin.photon.ml.TaskType
import com.linkedin.photon.ml.TaskType.TaskType
import com.linkedin.photon.ml.Types.{FeatureShardId, REType}
import com.linkedin.photon.ml.constants.DataConst
import com.linkedin.photon.ml.supervised.classification.{LogisticRegressionModel, SmoothedHingeLossLinearSVMModel}
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import com.linkedin.photon.ml.supervised.regression.{LinearRegressionModel, PoissonRegressionModel}
import com.linkedin.photon.ml.util.VectorUtils

/**
 * Representation of a random effect model.
 *
 * @param models The models, one for each unique random effect value
 * @param randomEffectType The random effect type
 * @param featureShardId The feature shard id
 */
class RandomEffectModel(
    val models: DataFrame,
    val randomEffectType: REType,
    val featureShardId: FeatureShardId)
  extends DatumScoringModel {

  override val modelType: TaskType = RandomEffectModel.determineModelType(models)

  //
  // RandomEffectModel functions
  //

  /**
   * Create a new [[RandomEffectModel]] with new underlying models.
   *
   * @param newModels The new underlying models, one per entity
   * @return A new [[RandomEffectModel]]
   */
  def update(newModels: DataFrame): RandomEffectModel =
    new RandomEffectModel(newModels, randomEffectType, featureShardId)

  //
  // DatumScoringModel functions
  //

  /**
   * Compute the score for the dataset.
   *
   * @note Use a static method to avoid serializing entire model object during RDD operations.
   * @param dataset The dataset to score
   * @return The computed scores
   */
  override def computeScore(dataset: DataFrame, scoreField: String): DataFrame = {

    RandomEffectModel.score(
      dataset,
      models,
      randomEffectType,
      featureShardId,
      scoreField)
  }

  //
  // Summarizable functions
  //

  /**
   * Summarize this model in text format.
   *
   * @return A model summary in String representation
   */
  override def toSummaryString: String = {

    val stringBuilder = new StringBuilder("Random Effect Model:")

    stringBuilder.append(s"\nRandom Effect Type: '$randomEffectType'")
    stringBuilder.append(s"\nFeature Shard ID: '$featureShardId'")
    //stringBuilder.append(s"\nLength: ${modelsRDD.values.map(_.coefficients.means.length).stats()}")
    //stringBuilder.append(s"\nMean: ${modelsRDD.values.map(_.coefficients.meansL2Norm).stats()}")
    //if (modelsRDD.first()._2.coefficients.variancesOption.isDefined) {
    //  stringBuilder.append(s"\nVariance: ${modelsRDD.values.map(_.coefficients.variancesL2NormOption.get).stats()}")
    //}

    stringBuilder.toString()
  }

  /**
   * Compares two [[RandomEffectModel]] objects.
   *
   * @param that Some other object
   * @return True if the models have the same types and the same underlying models for each random effect ID, false
   *         otherwise
   */
  override def equals(that: Any): Boolean = that match {

    case other: RandomEffectModel =>

      val areTypesEqual = this.randomEffectType == other.randomEffectType
      val areShardsEqual = this.featureShardId == other.featureShardId
      lazy val areAllModelsEqual = this
        .models
        .withColumnRenamed(DataConst.COEFFICIENTS, "s1")
        .join(other.models.withColumnRenamed(DataConst.COEFFICIENTS, "s2"), col(DataConst.ID), "fullouter")
        .filter("s1 is null or s2 is null or s1 != s2") //TODO: add udf to compare two vectors
        .head(1)
        .isEmpty

      areTypesEqual && areShardsEqual && areAllModelsEqual

    case _ =>
      false
  }

  /**
   * Convert models from dataframe to RDD
   * @return
   */
  def toRDD(): RDD[(REType, GeneralizedLinearModel)] = {
    models
      .select(randomEffectType, DataConst.MODEL_TYPE, DataConst.COEFFICIENTS)
      .rdd
      .map { row =>
        val reid = row.getInt(0).toString
        val modelType: TaskType = TaskType.withName(row.getString(1))
        val coefficients = Coefficients(VectorUtils.mlToBreeze(row.getAs[SparkVector](2)))

        val model = modelType match {
          case TaskType.LINEAR_REGRESSION =>
            LinearRegressionModel(coefficients)
          case TaskType.LOGISTIC_REGRESSION =>
            LogisticRegressionModel(coefficients)
          case TaskType.POISSON_REGRESSION =>
            PoissonRegressionModel(coefficients)
          case TaskType.SMOOTHED_HINGE_LOSS_LINEAR_SVM =>
            SmoothedHingeLossLinearSVMModel(coefficients)
        }
        (reid, model)
      }
  }

  /**
   * Returns a hash code value for the object.
   *
   * TODO: Violation of the hashCode() contract
   *
   * @return An [[Int]] hash code
   */
  override def hashCode(): Int = super.hashCode()

}

object RandomEffectModel {

  /**
   * Determine the random effect model type: even though the model has many sub-problems, there is only one loss
   * function type for a given random effect model.
   *
   * TODO: We should consider refactoring this method to instead take a TaskType and verify that all sub-models match
   *       that type - it will be faster for large numbers of random effect models. Note that it may still be a
   *       bottleneck if we check each time a new RandomEffectModel is created.
   *
   * @param models The random effect models
   * @return The GAME model type
   */
  protected def determineModelType(models: DataFrame): TaskType = {

    val modelTypes = models.select(GeneralizedLinearModel.MODEL_TYPE).head(1)

    require(
      modelTypes.length == 1,
      s"models has multiple model types:\n${modelTypes.mkString(", ")}")

    TaskType.withName(modelTypes(0).getString(0))
  }

  /**
   * Compute the scores for a dataset, using random effect models.
   *
   * @param dataset The dataset to score
   * @param models The individual random effect models to use for scoring
   * @param randomEffectType The random effect type
   * @param featureShardId The feature shard id
   * @return The scores
   */
  private def score (
      dataset: DataFrame,
      models: DataFrame,
      randomEffectType: REType,
      featureShardId: FeatureShardId,
      scoreField: String): DataFrame = {

    dataset
      .join(models, randomEffectType)
      .withColumn(scoreField, GeneralizedLinearModel.scoreUdf(col(DataConst.COEFFICIENTS), col(featureShardId)))
  }

  def toDataFrame(input: RDD[(REType, GeneralizedLinearModel)]): DataFrame = {
    null
  }
}
