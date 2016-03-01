package com.linkedin.photon.ml.util

import scala.collection.mutable

import breeze.linalg.{DenseVector, SparseVector, Vector, DenseMatrix, Matrix}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}

import com.linkedin.photon.ml.data.{GameData, KeyValueScore, LabeledPoint, LocalDataSet}
import com.linkedin.photon.ml.function.{LogisticLossFunction, SquaredLossFunction}
import com.linkedin.photon.ml.optimization.game.{
  GLMOptimizationConfiguration, MFOptimizationConfiguration, OptimizationProblem}
import com.linkedin.photon.ml.model.Coefficients
import com.linkedin.photon.ml.optimization.{LBFGS, TRON}
import com.linkedin.photon.ml.avro.data.NameAndTerm


/**
 * A collection of functions to configure the Spark context
 * @author xazhang
 */
object SparkContextConfiguration {

  /**
   * Configure the Spark context as a Yarn client
   * @param jobName The Spark application's name
   * @return The configured Spark context
   */
  def asYarnClient(jobName: String): SparkContext = {
    /* Configure the Spark application and initialize SparkContext, which is the entry point of a Spark application */
    val sparkConf = new SparkConf().setAppName(jobName)

    sparkConf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    sparkConf.registerKryoClasses(Array(
      classOf[mutable.BitSet],
      classOf[KeyValueScore],
      classOf[LogisticLossFunction],
      classOf[SquaredLossFunction],
      classOf[OptimizationProblem[_]],
      classOf[LocalDataSet],
      classOf[Coefficients],
      classOf[GLMOptimizationConfiguration],
      classOf[MFOptimizationConfiguration],
      classOf[NameAndTerm],
      classOf[LBFGS[LabeledPoint]],
      classOf[TRON[LabeledPoint]],
      classOf[LabeledPoint],
      classOf[Vector[Double]],
      classOf[SparseVector[Double]],
      classOf[DenseVector[Double]],
      classOf[Matrix[Double]],
      classOf[DenseMatrix[Double]],
      classOf[GameData]
    ))

    Logger.getRootLogger.setLevel(Level.WARN)
    new SparkContext(sparkConf)
  }
}
