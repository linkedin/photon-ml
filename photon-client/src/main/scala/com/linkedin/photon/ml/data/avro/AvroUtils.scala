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
package com.linkedin.photon.ml.data.avro

import java.lang.{Double => JDouble}
import java.util.{List => JList}

import scala.collection.JavaConversions._
import scala.collection.JavaConverters._
import scala.collection.{Set, mutable}
import scala.reflect.ClassTag

import breeze.linalg.{DenseVector, SparseVector, Vector}
import org.apache.avro.Schema
import org.apache.avro.Schema.Parser
import org.apache.avro.file.{DataFileStream, DataFileWriter}
import org.apache.avro.generic.{GenericDatumReader, GenericRecord}
import org.apache.avro.mapred._
import org.apache.avro.specific.{SpecificDatumReader, SpecificDatumWriter, SpecificRecord}
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.hadoop.io.NullWritable
import org.apache.hadoop.mapred.JobConf
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import com.linkedin.photon.avro.generated.{BayesianLinearModelAvro, LatentFactorAvro, NameTermValueAvro}
import com.linkedin.photon.ml.model.Coefficients
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import com.linkedin.photon.ml.util._

/**
 * Some basic functions to read/write Avro's [[GenericRecord]] from/to HDFS.
 */
object AvroUtils {

  // The upper limit of the file size the for reading single Avro file: 100 MB
  private val READ_SINGLE_AVRO_FILE_SIZE_LIMIT: Long = 100 << 20

  /**
   * Read Avro generic records from the input paths on HDFS.
   *
   * @note used in FeatureIndexJob, so can't be protected
   *
   * @param sc The Spark context
   * @param inputPaths The input paths to the generic records
   * @param minPartitions Minimum number of partitions of the output RDD
   * @return A [[RDD]] of Avro records of type [[GenericRecord]] read from the specified input paths
   */
  def readAvroFiles(
      sc: SparkContext,
      inputPaths: Seq[String],
      minPartitions: Int): RDD[GenericRecord] = {

    require(inputPaths.nonEmpty, "No input path specified - need at least 1")

    val minPartitionsPerPath = math.ceil(1.0 * minPartitions / inputPaths.length).toInt

    sc.union(inputPaths.map { path => readAvroFilesInDir[GenericRecord](sc, path, minPartitionsPerPath) } )
  }

  /**
   * Read all the Avro files in the given directory on HDFS.
   * The output RDD needs to be transformed before any further processing.
   *
   * @note used in GLMSuite, so can't be protected
   *
   * @param sc Spark context
   * @param inputDir The input directory to the Avro files
   * @param minNumPartitions Minimum number of Hadoop Splits to generate.
   * @return A RDD of records
   */
  def readAvroFilesInDir[T <: GenericRecord : ClassTag](
      sc: SparkContext,
      inputDir: String,
      minNumPartitions: Int): RDD[T] =
    sc.hadoopFile[AvroWrapper[T], NullWritable, AvroInputFormat[T]](inputDir, minNumPartitions)
      .map { case (k, _) => k.datum() }

  /**
   * Save an RDD of GenericRecord to HDFS using saveAsHadoopFile().
   *
   * @param data The data to write
   * @param outputDir The output directory to save the data as Avro files
   * @param schemaString The schema string of the data
   */
  protected[ml] def saveAsAvro[T <: SpecificRecord : ClassTag](data: RDD[T], outputDir: String, schemaString: String): Unit = {
    val job = new JobConf
    val schema: Schema = new Parser().parse(schemaString)
    AvroJob.setOutputSchema(job, schema)
    val dataFinal = data.map(row => (new AvroKey[T](row), NullWritable.get()))
    dataFinal.saveAsHadoopFile(
      outputDir,
      classOf[AvroKey[T]],
      classOf[NullWritable],
      classOf[AvroOutputFormat[T]],
      job
    )
  }

  /**
   * Read data from a single Avro file. It will return a list so do not use this method if data are large. According to
   * the class tag, this method will return generic or specific records.
   *
   * @note not protected because used in integTest training/DriverTest
   *
   * @param sc Spark context
   * @param path The path to a single Avro file (not the parent directory)
   * @param schemaString Optional schema string for reading
   * @tparam T The record type
   * @return List of records
   */
  def readFromSingleAvro[T <: GenericRecord : ClassTag](
    sc: SparkContext,
    path: String,
    schemaString: String = null): Seq[T] = {

    val classTag = implicitly[ClassTag[T]]
    val schema = if (schemaString == null) null else new Parser().parse(schemaString)
    val reader = if (classOf[SpecificRecord].isAssignableFrom(classTag.runtimeClass)) {
      new SpecificDatumReader[T](schema)
    } else {
      new GenericDatumReader[T](schema)
    }
    val fs = FileSystem.get(sc.hadoopConfiguration)
    val inputPath = new Path(path)
    val fileSize = fs.getContentSummary(inputPath).getLength

    require(fileSize < READ_SINGLE_AVRO_FILE_SIZE_LIMIT,
      s"The file $path is too large for readFromSingleAvro. Please use readFromAvro.")

    val inputStream = fs.open(inputPath)

    val dataFileReader = new DataFileStream[T](inputStream, reader)
    val it = dataFileReader.iterator()
    var buffer = Seq[T]()

    while (it.hasNext) {
      buffer = buffer :+ it.next
    }
    dataFileReader.close()
    buffer
  }

  /**
   * Write data to a single Avro file. It will only write to one Avro file so do not use this method if data are large.
   *
   * @param sc Spark context
   * @param data The Avro data to write
   * @param path The path to a single Avro file (not the parent directory)
   * @param schemaString The schema string
   * @param forceOverwrite Optional parameter to force overwrite
   * @tparam T The record type
   */
  protected[ml] def saveAsSingleAvro[T <: SpecificRecord : ClassTag](
    sc: SparkContext,
    data: Seq[T],
    path: String,
    schemaString: String,
    forceOverwrite: Boolean = false): Unit = {

    val outputPath = new Path(path)
    val fs = outputPath.getFileSystem(sc.hadoopConfiguration)
    val outputStream = fs.create(outputPath, forceOverwrite)
    val schema = new Parser().parse(schemaString)
    val writer = new SpecificDatumWriter[T](schema)
    val dataFileWriter = new DataFileWriter[T](writer)
    dataFileWriter.create(schema, outputStream)
    data.foreach(datum => dataFileWriter.append(datum))
    dataFileWriter.close()
  }

  /**
   * Convert the vector of type [[Vector[Double]] to an array of Avro records of type [[NameTermValueAvro]].
   *
   * @param vector The input vector
   * @param featureMap A map of feature index of type [[Int]] to feature name of type [[NameAndTerm]]
   * @return An array of Avro records that contains the information of the input vector
   */
  protected[avro] def convertVectorAsArrayOfNameTermValueAvros(
      vector: Vector[Double],
      featureMap: IndexMap): Array[NameTermValueAvro] =

    // TODO: Make vector sparsity configurable
    vector match {
      case dense: DenseVector[Double] =>
        dense
          .toArray
          .zipWithIndex
          .map(_.swap)
          .filter { case (_, value) =>
            math.abs(value) > VectorUtils.SPARSITY_THRESHOLD
          }
          .sortWith((p1, p2) => math.abs(p1._2) > math.abs(p2._2))
          .map { case (index, value) =>
            featureMap.getFeatureName(index) match {
              case Some(featureKey: String) =>
                val name = Utils.getFeatureNameFromKey(featureKey)
                val term = Utils.getFeatureTermFromKey(featureKey)

                NameTermValueAvro.newBuilder().setName(name).setTerm(term).setValue(value).build()

              case None =>
                throw new NoSuchElementException(s"Feature index $index not found in the feature map")
            }
          }

      case sparse: SparseVector[Double] =>
        sparse
          .activeIterator
          .filter { case (_, value) =>
            math.abs(value) > VectorUtils.SPARSITY_THRESHOLD
          }
          .toArray
          .sortWith((p1, p2) => math.abs(p1._2) > math.abs(p2._2))
          .map { case (index, value) =>
            featureMap.getFeatureName(index) match {
              case Some(featureKey: String) =>
                val name = Utils.getFeatureNameFromKey(featureKey)
                val term = Utils.getFeatureTermFromKey(featureKey)

                NameTermValueAvro.newBuilder().setName(name).setTerm(term).setValue(value).build()

              case None =>
                throw new NoSuchElementException(s"Feature index $index not found in the feature map")
            }
          }
    }

  /**
   * Read the nameAndTerm of type [[NameAndTerm]] from Avro record of type [[GenericRecord]].
   *
   * @param record The input Avro record
   * @return The nameAndTerm parsed from the Avro record
   */
  protected[avro] def readNameAndTermFromGenericRecord(record: GenericRecord): NameAndTerm = {
    val name = Utils.getStringAvro(record, AvroFieldNames.NAME)
    val term = Utils.getStringAvro(record, AvroFieldNames.TERM, isNullOK = true)
    NameAndTerm(name, term)
  }

  /**
   * Parse a set of nameAndTerm of type [[NameAndTerm]] from a RDD of Avro record of type [[GenericRecord]] with the
   * user specified feature section keys.
   *
   * @param genericRecords The input Avro records
   * @param featureSectionKey The user specified feature section keys
   * @return A set of nameAndTerms parsed from the input Avro records
   */
  protected[avro] def readNameAndTermSetFromGenericRecords(
      genericRecords: RDD[GenericRecord],
      featureSectionKey: String,
      numPartitions: Int): Set[NameAndTerm] = {

    genericRecords
      .flatMap {
        _.get(featureSectionKey) match {
          case recordList: JList[_] =>
            recordList.asScala.map {
              case record: GenericRecord =>
                AvroUtils.readNameAndTermFromGenericRecord(record)

              case any =>
                throw new IllegalArgumentException(
                  s"$any in features list is not a record. It needs to be an Avro record containingg a name and term for " +
                    s"each feature.")

            }
          case _ =>
            throw new IllegalArgumentException(
              s"$featureSectionKey is not a list (and might be null). It needs to be a list of Avro records containing a " +
                s"name and a term for each feature.")
        }
      }
      .distinct(numPartitions)
      .collect()
      .toSet
  }

  /**
   * Generate the [[NameAndTermFeatureSetContainer]] from a [[RDD]] of [[GenericRecord]]s.
   *
   * @param genericRecords The input [[RDD]] of [[GenericRecord]]s.
   * @param featureSectionKeys The set of feature section keys of interest in the input generic records
   * @return The generated [[NameAndTermFeatureSetContainer]]
   */
  protected[avro] def readNameAndTermFeatureSetContainerFromGenericRecords(
      genericRecords: RDD[GenericRecord],
      featureSectionKeys: Set[String],
      numPartitions: Int): NameAndTermFeatureSetContainer = {

    val nameAndTermFeatureSets = featureSectionKeys
      .map { featureSectionKey =>
        (featureSectionKey,
          AvroUtils.readNameAndTermSetFromGenericRecords(
            genericRecords,
            featureSectionKey,
            numPartitions))
      }
      .toMap

    new NameAndTermFeatureSetContainer(nameAndTermFeatureSets)
  }

  /**
   * Convert the coefficients of type [[Coefficients]] to Avro record of type [[BayesianLinearModelAvro]].
   *
   * @param modelId The model's id
   * @param featureMap The map from feature index of type [[Int]] to feature name of type [[NameAndTerm]]
   * @return The Avro record that contains the information of the input coefficients
   */
  protected[avro] def convertGLMModelToBayesianLinearModelAvro(
      model: GeneralizedLinearModel,
      modelId: String,
      featureMap: IndexMap): BayesianLinearModelAvro = {

    val modelCoefficients = model.coefficients
    val meansAvros = convertVectorAsArrayOfNameTermValueAvros(modelCoefficients.means, featureMap)
    val variancesAvrosOption = modelCoefficients
      .variancesOption
      .map(convertVectorAsArrayOfNameTermValueAvros(_, featureMap))
    // TODO: Output type of model.
    val avroFile = BayesianLinearModelAvro
      .newBuilder()
      .setModelId(modelId)
      .setModelClass(model.getClass.getName)
      .setLossFunction("")
      .setMeans(meansAvros.toList)

    if (variancesAvrosOption.isDefined) {
      avroFile.setVariances(variancesAvrosOption.get.toList)
    }

    avroFile.build()
  }

  /**
   * Convert the Avro record of type [[BayesianLinearModelAvro]] to the model type [[GeneralizedLinearModel]].
   *
   * @param bayesianLinearModelAvro The input Avro record
   * @param featureMap The map from feature name of type [[NameAndTerm]] to feature index of type [[Int]]
   * @return The generalized linear model converted from the Avro record
   */
  protected[avro] def convertBayesianLinearModelAvroToGLM(
      bayesianLinearModelAvro: BayesianLinearModelAvro,
      featureMap: IndexMap): GeneralizedLinearModel = {

    val meansAvros = bayesianLinearModelAvro.getMeans
    val modelClass = bayesianLinearModelAvro.getModelClass.toString
    val indexAndValueArrayBuffer = new mutable.ArrayBuffer[(Int, Double)]

    val iterator = meansAvros.iterator()
    while (iterator.hasNext) {
      val feature = iterator.next()
      val name = feature.getName.toString
      val term = feature.getTerm.toString
      val featureKey = Utils.getFeatureKey(name, term)
      if (featureMap.contains(featureKey)) {
        val value = feature.getValue
        val index = featureMap.getOrElse(featureKey,
          throw new NoSuchElementException(s"nameAndTerm $featureKey not found in the feature map"))
        indexAndValueArrayBuffer += ((index, value))
      }
    }

    val length = featureMap.featureDimension
    val coefficients = Coefficients(
      VectorUtils.toVector(indexAndValueArrayBuffer.toArray, length))

    // Load and instantiate the model
    try {
      Class.forName(modelClass)
        .getConstructor(classOf[Coefficients])
        .newInstance(coefficients)
        .asInstanceOf[GeneralizedLinearModel]

    } catch {
      case e: Exception =>
        throw new IllegalArgumentException(
          s"Error loading model: model class $modelClass couldn't be loaded. You may need to retrain the model.", e)
    }
  }

  /**
   * Convert the latent factor of type [[Vector[Double]]] to Avro record of type [[LatentFactorAvro]].
   *
   * @param effectId The id of the latent factor, e.g., row Id, col Id, user Id or itemId
   * @param latentFactor The latent factor of the matrix factorization model
   * @return The Avro record that contains the information of the input latent factor
   */
  protected[avro] def convertLatentFactorToLatentFactorAvro(
      effectId: String,
      latentFactor: Vector[Double]): LatentFactorAvro = {

    val latentFactorAsList = latentFactor.toArray.map(JDouble.valueOf).toList
    val avroFile = LatentFactorAvro.newBuilder().setEffectId(effectId).setLatentFactor(latentFactorAsList)
    avroFile.build()
  }

  /**
   * Convert the given Avro record of type [[LatentFactorAvro]] to the latent factor of type [[Vector[Double]].
   *
   * @param latentFactorAvro The given Avro record
   * @return The (effectId, latentFactor) pair converted from the input Avro record
   */
  protected[avro] def convertLatentFactorAvroToLatentFactor(
      latentFactorAvro: LatentFactorAvro): (String, Vector[Double]) = {

    val effectId = latentFactorAvro.getEffectId.toString
    val latentFactor = new DenseVector[Double](latentFactorAvro.getLatentFactor.toArray().map(_.asInstanceOf[Double]))
    (effectId, latentFactor)
  }

  /**
   * Creates a default map of features, i.e. a Map[(String, Int)] from feature names to an ordinal counter,
   * by scanning through a single model (typically, the fixed effects model).
   * This is useful e.g. in model format conversions when we don't have the data to go over to build a
   * feature index map.
   *
   * @note There might be faster ways to implement this: all the feature names are gathered first, then duplicates are
   *       removed, when duplicates could be removed while aggregating the names, probably consuming a lot less memory.
   *
   * @param modelAvro The model (only one) to scan for feature names
   * @return A DefaultIndexMap, which associates a unique integer id to each feature name
   */
  protected[avro] def makeFeatureIndexForModel(modelAvro: BayesianLinearModelAvro): IndexMap =
    DefaultIndexMap(
      modelAvro.getMeans.map(nameTermValue => Utils.getFeatureKey(nameTermValue.getName, nameTermValue.getTerm)))

  /**
   * Creates a default map of features, i.e. a Map[(String, Int)] from feature names to an ordinal counter,
   * by scanning through multiple models (typically, several random effects models).
   * This is useful e.g. in model format conversions when we don't have the data to go over to build a
   * feature index map.
   *
   * @note There might be faster ways to implement this: all the feature names are gathered first, then duplicates are
   *       removed, when duplicates could be removed while aggregating the names, probably consuming a lot less memory.
   *
   * @param sc The Spark context
   * @param modelAvros The models to scan for feature names
   * @return A DefaultIndexMapLoader, which associates a unique integer id to each feature name
   */
  protected[avro] def makeFeatureIndexForModel(
      sc: SparkContext,
      modelAvros: RDD[BayesianLinearModelAvro]): IndexMapLoader = {

    DefaultIndexMapLoader(
      sc,
      modelAvros
        .flatMap(modelAvro =>
          modelAvro
            .getMeans
            .map(nameTermValue => Utils.getFeatureKey(nameTermValue.getName, nameTermValue.getTerm)))
        .collect)
  }
}
