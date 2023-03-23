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
package com.linkedin.photon.ml

import scala.collection.mutable

import breeze.linalg.{DenseMatrix, DenseVector, Matrix, SparseVector, Vector}
import org.apache.spark.serializer.KryoSerializer
import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkConf

import com.linkedin.photon.ml.aggregators._
import com.linkedin.photon.ml.data.{GameDatum, LabeledPoint, LocalDataset}
import com.linkedin.photon.ml.function._
import com.linkedin.photon.ml.model.Coefficients
import com.linkedin.photon.ml.normalization.NormalizationContext
import com.linkedin.photon.ml.optimization._
import com.linkedin.photon.ml.optimization.game.GLMOptimizationConfiguration
import com.linkedin.photon.ml.projector.LinearSubspaceProjector
import com.linkedin.photon.ml.supervised.classification.LogisticRegressionModel
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import com.linkedin.photon.ml.supervised.regression.{LinearRegressionModel, PoissonRegressionModel}

/**
 * Factory for creating [[SparkSession]] instances. This handles the tricky details of things like setting up
 * serialization, resource negotiation, logging, etc.
 */
object SparkSessionConfiguration {

  val CONF_SPARK_APP_NAME = "spark.app.name"
  val CONF_SPARK_SERIALIZER = "spark.serializer"
  val CONF_SPARK_KRYO_CLASSES_TO_REGISTER = "spark.kryo.classesToRegister"
  val KRYO_CLASSES_TO_REGISTER: Array[Class[_]] = Array(
    classOf[mutable.BitSet],
    classOf[mutable.ListBuffer[_]],
    classOf[Coefficients],
    classOf[DenseMatrix[Double]],
    classOf[DenseVector[Double]],
    classOf[GameDatum],
    classOf[GeneralizedLinearModel],
    classOf[GeneralizedLinearOptimizationProblem[_]],
    classOf[GLMOptimizationConfiguration],
    classOf[HessianDiagonalAggregator],
    classOf[HessianMatrixAggregator],
    classOf[HessianVectorAggregator],
    classOf[LinearSubspaceProjector],
    classOf[LabeledPoint],
    classOf[LBFGS],
    classOf[LinearRegressionModel],
    classOf[LocalDataset],
    classOf[LogisticRegressionModel],
    classOf[Matrix[Double]],
    classOf[NormalizationContext],
    classOf[ObjectiveFunction],
    classOf[OptimizationStatesTracker],
    classOf[Option[_]],
    classOf[OWLQN],
    classOf[PoissonRegressionModel],
    classOf[RegularizationContext],
    classOf[Set[Int]],
    classOf[SingleNodeObjectiveFunction],
    classOf[SingleNodeOptimizationProblem[_]],
    classOf[SparseVector[Double]],
    classOf[TRON],
    classOf[ValueAndGradientAggregator],
    classOf[Vector[Double]])

  /**
   * Configure the Spark context as a Yarn client.
   *
   * @param sparkConf The Spark Conf object
   * @param jobName The Spark application's name
   * @param useKryo Whether to use kryo to serialize RDD and intermediate data
   * @return The configured Spark context
   */
  def asYarnClient(sparkConf: SparkConf, jobName: String, useKryo: Boolean): SparkSession = {

    // Configure the Spark application and initialize SparkContext, which is the entry point of a Spark application
    sparkConf.setAppName(jobName)

    if (useKryo) {
      sparkConf.set(CONF_SPARK_SERIALIZER, classOf[KryoSerializer].getName)
      sparkConf.registerKryoClasses(KRYO_CLASSES_TO_REGISTER)
    }

    SparkSession.builder.config(sparkConf).getOrCreate()
  }

  /**
   * Configure the Spark context as a Yarn client.
   *
   * @param jobName The Spark application's name
   * @param useKryo Whether to use kryo to serialize RDD and intermediate data
   * @return The configured Spark context
   */
  def asYarnClient(jobName: String, useKryo: Boolean): SparkSession =
    asYarnClient(new SparkConf(), jobName, useKryo)
}
