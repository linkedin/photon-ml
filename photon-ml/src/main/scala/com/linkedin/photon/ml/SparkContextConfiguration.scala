package com.linkedin.photon.ml

import breeze.linalg.{DenseVector, SparseVector, Vector}
import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.function.{
  GeneralizedLinearModelLossFunction, HessianVectorAggregator, ValueAndGradientAggregator}
import com.linkedin.photon.ml.normalization.NormalizationContext
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
  val KRYO_CLASSES_TO_REGISTER = Array[Class[_]](classOf[LabeledPoint],
    classOf[Vector[Double]],
    classOf[SparseVector[Double]],
    classOf[DenseVector[Double]],
    classOf[Set[Int]],
    classOf[ValueAndGradientAggregator],
    classOf[HessianVectorAggregator],
    classOf[GeneralizedLinearModelLossFunction],
    classOf[NormalizationContext])

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
}
