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
package com.linkedin.photon.ml.avro

import scala.reflect.ClassTag

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

/**
 * Utility to read/write Avro files.
 */
object AvroIOUtils {

  // The upper limit of the file size the for reading single Avro file: 100 MB
  private val READ_SINGLE_AVRO_FILE_SIZE_LIMIT: Long = 100 << 20

  /**
   * Read Avro generic records from the input paths on HDFS.
   *
   * @param sc The Spark context
   * @param inputPaths The input paths to the generic records
   * @param minPartitions Minimum number of partitions of the output RDD
   * @return A [[RDD]] of Avro records of type [[GenericRecord]] read from the specified input paths
   */
  protected[ml] def readAvroFiles(
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
      .map  { case (k, _) => k.datum() }

  /**
   * Save an RDD of GenericRecord to HDFS using saveAsHadoopFile().
   *
   * @param data The data to write
   * @param outputDir The output directory to save the data as Avro files
   * @param schemaString The schema string of the data
   */
  def saveAsAvro[T <: SpecificRecord : ClassTag](data: RDD[T], outputDir: String, schemaString: String): Unit = {
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
  def saveAsSingleAvro[T <: SpecificRecord : ClassTag](
      sc: SparkContext,
      data: Seq[T],
      path: String,
      schemaString: String,
      forceOverwrite: Boolean = false): Unit = {

    val fs = FileSystem.get(sc.hadoopConfiguration)
    val outputPath = new Path(path)
    val outputStream = fs.create(outputPath, forceOverwrite)
    val schema = new Parser().parse(schemaString)
    val writer = new SpecificDatumWriter[T](schema)
    val dataFileWriter = new DataFileWriter[T](writer)
    dataFileWriter.create(schema, outputStream)
    data.foreach(datum => dataFileWriter.append(datum))
    dataFileWriter.close()
  }
}
