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
package com.linkedin.photon.ml

import breeze.linalg.{DenseMatrix, DenseVector, Matrix, SparseVector, Vector}
import scala.collection.mutable.BitSet

import com.linkedin.photon.ml.avro.data.NameAndTerm
import com.linkedin.photon.ml.data.{GameData, KeyValueScore, LabeledPoint, LocalDataSet}
import com.linkedin.photon.ml.function.{
  LogisticLossFunction, SquaredLossFunction, GeneralizedLinearModelLossFunction, HessianVectorAggregator,
  ValueAndGradientAggregator}
import com.linkedin.photon.ml.model.Coefficients
import com.linkedin.photon.ml.normalization.NormalizationContext
import com.linkedin.photon.ml.optimization.{LBFGS, TRON}
import com.linkedin.photon.ml.optimization.game.{
  GLMOptimizationConfiguration, MFOptimizationConfiguration, OptimizationProblem}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.serializer.KryoSerializer
import org.apache.spark.{SparkConf, SparkContext}

/**
 * Factory for creating SparkContext instances. This handles the tricky details of things like setting up serialization,
 * resource negotiation, logging, etc.
 *
 * @author xazhang
 * @author yizhou
 */
object SparkContextConfiguration {
  val CONF_SPARK_APP_NAME = "spark.app.name"
  val CONF_SPARK_SERIALIZER = "spark.serializer"
  val CONF_SPARK_KRYO_CLASSES_TO_REGISTER = "spark.kryo.classesToRegister"
  val KRYO_CLASSES_TO_REGISTER = Array[Class[_]](
    classOf[BitSet],
    classOf[Coefficients],
    classOf[DenseMatrix[Double]],
    classOf[DenseVector[Double]],
    classOf[GLMOptimizationConfiguration],
    classOf[GameData],
    classOf[GeneralizedLinearModelLossFunction],
    classOf[HessianVectorAggregator],
    classOf[KeyValueScore],
    classOf[LBFGS[LabeledPoint]],
    classOf[LabeledPoint],
    classOf[LocalDataSet],
    classOf[LogisticLossFunction],
    classOf[MFOptimizationConfiguration],
    classOf[Matrix[Double]],
    classOf[NameAndTerm],
    classOf[NormalizationContext],
    classOf[OptimizationProblem[_]],
    classOf[Set[Int]],
    classOf[SparseVector[Double]],
    classOf[SquaredLossFunction],
    classOf[TRON[LabeledPoint]],
    classOf[ValueAndGradientAggregator],
    classOf[Vector[Double]])

  /**
   * Configure the Spark context as a Yarn client
   *
   * @param sparkConf The Spark Conf object
   * @param jobName The Spark application's name
   * @param useKryo Whether to use kryo to serialize RDD and intermediate data
   * @return The configured Spark context
   */
  def asYarnClient(sparkConf: SparkConf, jobName: String, useKryo: Boolean): SparkContext = {
    /* Configure the Spark application and initialize SparkContext, which is the entry point of a Spark application */
    sparkConf.setAppName(jobName)
    if (useKryo) {
      sparkConf.set(CONF_SPARK_SERIALIZER, classOf[KryoSerializer].getName)
      sparkConf.registerKryoClasses(KRYO_CLASSES_TO_REGISTER)
    }
    Logger.getRootLogger.setLevel(Level.INFO)
    new SparkContext(sparkConf)
  }

  /**
   * Configure the Spark context as a Yarn client
   *
   * @param jobName The Spark application's name
   * @param useKryo Whether to use kryo to serialize RDD and intermediate data
   * @return The configured Spark context
   */
  def asYarnClient(jobName: String, useKryo: Boolean): SparkContext = {
    val sparkConf = new SparkConf()
    asYarnClient(sparkConf, jobName, useKryo)
  }
}
