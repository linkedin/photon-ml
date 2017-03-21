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

import breeze.linalg.{Vector, norm}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import com.linkedin.photon.ml.TaskType
import com.linkedin.photon.ml.TaskType.TaskType
import com.linkedin.photon.ml.data.GameDatum
import com.linkedin.photon.ml.data.scoring.ModelDataScores
import com.linkedin.photon.ml.spark.RDDLike

/**
 * Representation of a matrix factorization model.
 *
 * @param rowEffectType What each row of the matrix corresponds to, e.g., memberId or itemId
 * @param colEffectType What each column of the matrix corresponds to, e.g., memberId or itemId
 * @param rowLatentFactors Latent factors for row effect
 * @param colLatentFactors Latent factors for column effect
 */
class MatrixFactorizationModel(
    val rowEffectType: String,
    val colEffectType: String,
    val rowLatentFactors: RDD[(String, Vector[Double])],
    val colLatentFactors: RDD[(String, Vector[Double])]) extends DatumScoringModel with RDDLike {

  // TODO: This will need to change to play nicely with the other DatumScoringModels as part of a single GAMEModel.
  val modelType: TaskType = TaskType.NONE

  /**
   * Number of latent factors of the matrix factorization model (or the rank)
   */
  lazy val numLatentFactors =
    if (!rowLatentFactors.isEmpty()) {
      rowLatentFactors.first()._2.length
    } else if (!colLatentFactors.isEmpty()) {
      colLatentFactors.first()._2.length
    } else {
      0
    }

  /**
   * Score the effect-specific data set in the coordinate with the input model.
   *
   * @param dataPoints The dataset to score. Note that the Long in the RDD is a unique identifier for the paired
   *                   GAMEDatum object, referred to in the GAME code as the "unique id".
   * @return The score.
   */
  override def score(dataPoints: RDD[(Long, GameDatum)]): ModelDataScores =
    MatrixFactorizationModel.score(dataPoints, rowEffectType, colEffectType, rowLatentFactors, colLatentFactors)

  /**
   * Build a human-readable summary for this [[MatrixFactorizationModel]].
   *
   * @return A summary of the [[MatrixFactorizationModel]] in string representation
   */
  override def toSummaryString: String = {
    val stringBuilder = new StringBuilder(s"Summary of matrix factorization model with rowEffectType $rowEffectType " +
      s"and colEffectType $colEffectType:")
    val rowLatentFactorsL2NormStats =
      rowLatentFactors.map { case (_, rowLatentFactor) => norm(rowLatentFactor, 2) } .stats()
    val colLatentFactorsL2NormStats =
      colLatentFactors.map { case (_, colLatentFactor) => norm(colLatentFactor, 2) } .stats()
    stringBuilder.append(s"\nnumLatentFactors: $numLatentFactors")
    stringBuilder.append(s"\nrowLatentFactors L2 norm: $rowLatentFactorsL2NormStats")
    stringBuilder.append(s"\ncolLatentFactors L2 norm: $colLatentFactorsL2NormStats")
    stringBuilder.toString()
  }

  /**
   * Compares two [[MatrixFactorizationModel]] objects.
   *
   * @param that Some other object
   * @return True if both models have the same row and column latent factors, false otherwise
   */
  override def equals(that: Any): Boolean = {
    that match {
      case other: MatrixFactorizationModel =>
        val sameMetaData = this.rowEffectType == other.rowEffectType && this.colEffectType == other.colEffectType &&
          this.numLatentFactors == other.numLatentFactors
        val sameRowLatentFactors =
          MatrixFactorizationModel.isSameLatentFactors(this.rowLatentFactors, other.rowLatentFactors)
        val sameColLatentFactors =
          MatrixFactorizationModel.isSameLatentFactors(this.colLatentFactors, other.colLatentFactors)
        sameMetaData && sameRowLatentFactors && sameColLatentFactors
      case _ => false
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

  /**
   * Get the Spark context.
   *
   * @return The Spark context
   */
  override def sparkContext: SparkContext = rowLatentFactors.sparkContext

  /**
   * Assign a given name to [[rowLatentFactors]] and [[colLatentFactors]].
   *
   * @note Not used to reference models in the logic of photon-ml, only used for logging currently.
   *
   * @param name The parent name for all [[RDD]]s in this class
   * @return This object with the names of [[rowLatentFactors]] and [[colLatentFactors]] assigned
   */
  override def setName(name: String): MatrixFactorizationModel = {
    rowLatentFactors.setName(s"$name: row latent factors")
    colLatentFactors.setName(s"$name: col latent factors")
    this
  }

  /**
   * Set the storage level of [[rowLatentFactors]] and [[colLatentFactors]], and persist their values across the cluster
   * the first time they are computed.
   *
   * @param storageLevel The storage level
   * @return This object with the storage level of [[rowLatentFactors]] and [[colLatentFactors]] set
   */
  override def persistRDD(storageLevel: StorageLevel): MatrixFactorizationModel = {
    if (!rowLatentFactors.getStorageLevel.isValid) rowLatentFactors.persist(storageLevel)
    if (!colLatentFactors.getStorageLevel.isValid) colLatentFactors.persist(storageLevel)
    this
  }

  /**
   * Mark [[rowLatentFactors]] and [[colLatentFactors]] as non-persistent, and remove all blocks for them from memory
   * and disk.
   *
   * @return This object with [[rowLatentFactors]] and [[colLatentFactors]] marked non-persistent
   */
  override def unpersistRDD(): MatrixFactorizationModel = {
    if (rowLatentFactors.getStorageLevel.isValid) rowLatentFactors.unpersist()
    if (colLatentFactors.getStorageLevel.isValid) colLatentFactors.unpersist()
    this
  }

  /**
   * Materialize [[rowLatentFactors]] and [[colLatentFactors]] (Spark [[RDD]]s are lazy evaluated: this method forces
   * them to be evaluated).
   *
   * @return This object with [[rowLatentFactors]] and [[colLatentFactors]] materialized
   */
  override def materialize(): MatrixFactorizationModel = {
    rowLatentFactors.count()
    colLatentFactors.count()
    this
  }
}

object MatrixFactorizationModel {

  /**
   * Check whether two latent factors are the same.
   *
   * @param latentFactors1 The first latent factor
   * @param latentFactors2 The second latent factor
   * @return True if the two latent factors are the same, false otherwise
   */
  protected def isSameLatentFactors(
    latentFactors1: RDD[(String, Vector[Double])],
    latentFactors2: RDD[(String, Vector[Double])]): Boolean = {
    latentFactors1.fullOuterJoin(latentFactors2).mapPartitions(iterator =>
      Iterator.single(iterator.forall { case (_, (factor1, factor2)) =>
        factor1.isDefined && factor2.isDefined && factor1.get.equals(factor2.get)
      })
    ).filter(!_).count() == 0
  }

  /**
   * Score the given GAME data points with the row and column latent factors.
   *
   * @param dataPoints The GAME data points
   * @param rowEffectType The type of row effect used to retrieve row effect id from each GAME data point
   * @param colEffectType The type of column effect used to retrieve column effect id from each GAME data point
   * @param rowLatentFactors The row latent factors
   * @param colLatentFactors The col latent factors
   * @return The computed scores
   */
  protected def score(
      dataPoints: RDD[(Long, GameDatum)],
      rowEffectType: String,
      colEffectType: String,
      rowLatentFactors: RDD[(String, Vector[Double])],
      colLatentFactors: RDD[(String, Vector[Double])]): ModelDataScores =
    new ModelDataScores(dataPoints
      .map { case (uniqueId, gameDatum) =>
        // For each datum, collect a (rowEffectId, (colEffectId, uniqueId)) tuple.
        val rowEffectId = gameDatum.idTypeToValueMap(rowEffectType)
        val colEffectId = gameDatum.idTypeToValueMap(colEffectType)
        (rowEffectId, (colEffectId, uniqueId, gameDatum.toScoredGameDatum()))
      }
      .cogroup(rowLatentFactors)
      .flatMap { case (rowEffectId, (colEffectIdAndUniqueIdsAndDatumIterable, rowLatentFactorIterable)) =>
        // Decorate rowEffectId with row latent factors
        assert(rowLatentFactorIterable.size <= 1, s"More than one row latent factor (${rowLatentFactorIterable.size}) "
          + s"found for random effect id $rowEffectId of random effect type $rowEffectType")
        colEffectIdAndUniqueIdsAndDatumIterable.flatMap { case (colEffectId, uniqueId, scoredDatum) =>
          rowLatentFactorIterable.map(rowLatentFactor => (colEffectId, (rowLatentFactor, uniqueId, scoredDatum)))
        }
      }
      .cogroup(colLatentFactors)
      .flatMap { case (colEffectId, (rowLatentFactorAndUniqueIdsAndDatumIterable, colLatentFactorIterable)) =>
        // Decorate colEffectId with column latent factors
        val size = colLatentFactorIterable.size
        assert(size <= 1, s"More than one column latent factor ($size) found for random effect id $colEffectId of "
          + s"random effect type $colEffectType")
        rowLatentFactorAndUniqueIdsAndDatumIterable.flatMap { case (rowLatentFactor, uniqueId, scoredDatum) =>
          colLatentFactorIterable.map { colLatentFactor =>
            (uniqueId, scoredDatum.copy(score = rowLatentFactor.dot(colLatentFactor)))
          }
        }
      })
}
