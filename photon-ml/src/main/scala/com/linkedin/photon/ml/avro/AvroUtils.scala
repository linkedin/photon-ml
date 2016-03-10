package com.linkedin.photon.ml.avro


import scala.collection.JavaConversions._
import scala.collection.Map

import breeze.linalg.{DenseVector, SparseVector, Vector}
import org.apache.avro.generic.GenericRecord
import org.apache.avro.mapred.{AvroInputFormat, AvroWrapper}
import org.apache.hadoop.io.NullWritable
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.avro.generated.{BayesianLinearModelAvro, NameTermValueAvro}
import com.linkedin.photon.ml.avro.data.NameAndTerm
import com.linkedin.photon.ml.constants.{AvroFieldNames, MathConst}
import com.linkedin.photon.ml.model.Coefficients
import com.linkedin.photon.ml.util.Utils


/**
 * @author xazhang
 */
protected[photon] object AvroUtils {

  /**
   * Read Avro files from a directory
   */
  def readAvroFiles(sc: SparkContext,
                    inputPaths: Seq[String],
                    minPartitions: Int): RDD[GenericRecord] = {
    assert(inputPaths.nonEmpty, "The number of input paths is zero.")
    val minPartitionsPerPath = math.ceil(1.0 * minPartitions / inputPaths.length).toInt
    inputPaths.map { path =>
      sc.hadoopFile[AvroWrapper[GenericRecord], NullWritable, AvroInputFormat[GenericRecord]](path,
        minPartitionsPerPath)
    }.reduce(_ ++ _).map(_._1.datum())
  }

  private def vectorToArrayOfNameTermValueAvros(vector: Vector[Double], featureMap: Map[Int, NameAndTerm])
  : Array[NameTermValueAvro] = {

    vector match {
      case dense: DenseVector[Double] =>
        dense.toArray.zipWithIndex.map(_.swap).filter { case (key, value) =>
          math.abs(value) > MathConst.LOW_PRECISION_TOLERANCE_THRESHOLD
        }
          .sortWith((p1, p2) => math.abs(p1._2) > math.abs(p2._2)).map { case (index, value) =>
          featureMap.get(index) match {
            case Some(NameAndTerm(name, term)) =>
              NameTermValueAvro.newBuilder().setName(name).setTerm(term).setValue(value).build()
            case None =>
              throw new NoSuchElementException(s"Feature index $index not found in the feature map")
          }
        }
      case sparse: SparseVector[Double] =>
        sparse.activeIterator.filter { case (key, value) =>
          math.abs(value) > MathConst.LOW_PRECISION_TOLERANCE_THRESHOLD
        }.toArray
          .sortWith((p1, p2) => math.abs(p1._2) > math.abs(p2._2)).map { case (index, value) =>
          featureMap.get(index) match {
            case Some(NameAndTerm(name, term)) =>
              NameTermValueAvro.newBuilder().setName(name).setTerm(term).setValue(value).build()
            case None =>
              throw new NoSuchElementException(s"Feature index $index not found in the feature map")
          }
        }
    }
  }

  def getNameAndTermFromAvroRecord(record: GenericRecord): NameAndTerm = {
    val name = Utils.getStringAvro(record, AvroFieldNames.NAME, isNullOK = false)
    val term = Utils.getStringAvro(record, AvroFieldNames.TERM, isNullOK = false)
    NameAndTerm(name, term)
  }

  def modelToBayesianLinearModelAvro(
      model: Coefficients,
      modelId: String,
      intToNameAndTermMap: Map[Int, NameAndTerm]): BayesianLinearModelAvro = {
    val meansAvros = vectorToArrayOfNameTermValueAvros(model.means, intToNameAndTermMap)
    val variancesAvrosOption = model.variancesOption.map(variances => vectorToArrayOfNameTermValueAvros(variances,
      intToNameAndTermMap))
    val avroFile = BayesianLinearModelAvro.newBuilder().setModelId(modelId).setLossFunction("")
        .setMeans(meansAvros.toList)
    if (variancesAvrosOption.isDefined) avroFile.setVariances(variancesAvrosOption.get.toList)
    avroFile.build()
  }

  // Here we only load means
  def loadMeanVectorFromBayesianLinearModelAvro(
      bayesianLinearModelAvro: BayesianLinearModelAvro,
      nameAndTermToIntMap: Map[NameAndTerm, Int]): Vector[Double] = {
    val meansAvros = bayesianLinearModelAvro.getMeans
    val indexAndValueArray = new Array[Double](nameAndTermToIntMap.size)
    val iterator = meansAvros.iterator()
    while (iterator.hasNext) {
      val feature = iterator.next()
      val name = feature.getName.toString
      val term = feature.getTerm.toString
      val nameAndTerm = NameAndTerm(name, term)
      if (nameAndTermToIntMap.contains(nameAndTerm)) {
        val value = feature.getValue
        val index = nameAndTermToIntMap.getOrElse(nameAndTerm,
          throw new NoSuchElementException(s"nameAndTerm $nameAndTerm not found in the feature map"))
        indexAndValueArray(index) = value
      }
    }
    new DenseVector(indexAndValueArray)
  }
}
