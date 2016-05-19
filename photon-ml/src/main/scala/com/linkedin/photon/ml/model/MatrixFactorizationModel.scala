/*
 * Copyright 2016 LinkedIn Corp. All rights reserved.
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
import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.data.{GameDatum, KeyValueScore}


/**
 * Representation of a matrix factorization model
 * @param rowEffectType What each row of the matrix corresponds to, e.g., memberId or itemId
 * @param colEffectType What each column of the matrix corresponds to, e.g., memberId or itemId
 * @param rowLatentFactors Latent factors for row effect
 * @param colLatentFactors Latent factors for column effect
 */
class MatrixFactorizationModel(
  val rowEffectType: String,
  val colEffectType: String,
  val rowLatentFactors: RDD[(String, Vector[Double])],
  val colLatentFactors: RDD[(String, Vector[Double])]) extends Model {

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

  override def score(dataPoints: RDD[(Long, GameDatum)]): KeyValueScore = {
    MatrixFactorizationModel.score(dataPoints, rowEffectType, colEffectType, rowLatentFactors, colLatentFactors)
  }

  override def toSummaryString: String = {
    val stringBuilder = new StringBuilder(s"Summary of matrix factorization model with rowEffectType $rowEffectType " +
      s"and colEffectType $colEffectType:")
    val rowLatentFactorsL2NormStats =
      rowLatentFactors.map { case (_, rowLatentFactor) => norm(rowLatentFactor, 2) } .stats()
    val colLatentFactorsL2NormStats =
      colLatentFactors.map { case (_, colLatentFactor) => norm(colLatentFactor, 2) } .stats()
    stringBuilder.append(s"\numLatentFactors: $numLatentFactors")
    stringBuilder.append(s"\nrowLatentFactors L2 norm: $rowLatentFactorsL2NormStats")
    stringBuilder.append(s"\ncolLatentFactors L2 norm: $colLatentFactorsL2NormStats")
    stringBuilder.toString()
  }

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

  override def hashCode(): Int = {
    super.hashCode()
  }
}

object MatrixFactorizationModel {

  protected def isSameLatentFactors(
    latentFactors1: RDD[(String, Vector[Double])],
    latentFactors2: RDD[(String, Vector[Double])]): Boolean = {
    latentFactors1.fullOuterJoin(latentFactors2).mapPartitions(iterator =>
      Iterator.single(iterator.forall { case (_, (factor1, factor2)) =>
        factor1.isDefined && factor2.isDefined && factor1.get.equals(factor2.get)
      })
    ).filter(!_).count() == 0
  }

  protected def score(
    dataPoints: RDD[(Long, GameDatum)],
    rowEffectType: String,
    colEffectType: String,
    rowLatentFactors: RDD[(String, Vector[Double])],
    colLatentFactors: RDD[(String, Vector[Double])]): KeyValueScore = {

    val scores = dataPoints
      //For each datum, collect a (rowEffectId, (colEffectId, uniqueId)) tuple.
      .map { case (uniqueId, gameData) =>
      val rowEffectId = gameData.randomEffectIdToIndividualIdMap(rowEffectType)
      val colEffectId = gameData.randomEffectIdToIndividualIdMap(colEffectType)
      (rowEffectId, (colEffectId, uniqueId))
    }
      .cogroup(rowLatentFactors)
      // Decorate rowEffectId with row latent factors
      .flatMap { case (rowEffectId, (colEffectIdAndUniqueIdsIterable, rowLatentFactorIterable)) =>
      assert(rowLatentFactorIterable.size <= 1, s"More than one row latent factor (${rowLatentFactorIterable.size}) " +
        s"found for random effect Id $rowEffectId of random effect type $rowEffectType")
      colEffectIdAndUniqueIdsIterable.flatMap { case (colEffectId, uniqueId) =>
        rowLatentFactorIterable.map(rowLatentFactor => (colEffectId, (rowLatentFactor, uniqueId)))
      }
    }
      // Decorate colEffectId with column latent factors
      .cogroup(colLatentFactors)
      .flatMap { case (colEffectId, (rowLatentFactorAndUniqueIdsIterable, colLatentFactorIterable)) =>
        val size = colLatentFactorIterable.size
      assert(size <= 1, s"More than one column latent factor ($size) found for random effect Id $colEffectId of " +
        s"random effect type $colEffectType")
      rowLatentFactorAndUniqueIdsIterable.flatMap { case (rowLatentFactor, uniqueId) =>
        colLatentFactorIterable.map(colLatentFactor => (uniqueId, (rowLatentFactor, colLatentFactor)))
      }
    }
      // Compute dot product for (row latent factors, column latent factors) tuple
      .mapValues { case (rowLatentFactor, colLatentFactor) => rowLatentFactor.dot(colLatentFactor) }

    new KeyValueScore(scores)
  }
}
