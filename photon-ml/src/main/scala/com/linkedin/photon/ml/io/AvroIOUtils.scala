package com.linkedin.photon.ml.io

import org.apache.avro.Schema
import org.apache.avro.Schema.Parser
import org.apache.avro.file.{DataFileStream, DataFileWriter}
import org.apache.avro.generic.{GenericDatumReader, GenericDatumWriter, GenericRecord}
import org.apache.avro.mapred._
import org.apache.avro.specific.{SpecificDatumReader, SpecificRecord}
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.hadoop.io.NullWritable
import org.apache.hadoop.mapred.JobConf
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

/**
 * Utility to write Avro files
 */

object AvroIOUtils {
  // The upper limit of the file size the for reading single Avro file: 100 MB
  private val READ_SINGLE_AVRO_FILE_SIZE_LIMIT: Long = 100 << 20

  /**
   * Read GenericRecord. The output RDD needs to be transformed before any further processing.
   * @param sc Spark context
   * @param inputDir The input directory to the Avro files
   * @param minNumPartitions Minimum number of Hadoop Splits to generate.
   * @return A RDD of records
   */
  def readFromAvro[T <: GenericRecord : ClassTag](sc: SparkContext, inputDir: String, minNumPartitions: Int): RDD[T] = {
    sc.hadoopFile[AvroWrapper[T], NullWritable, AvroInputFormat[T]](inputDir, minNumPartitions).map({
      case (k, v) => k.datum()
    })
  }

  /**
   * Save an RDD of GenericRecord to HDFS using saveAsHadoopFile()
   * @param data The data to write
   * @param outputDir The output directory to save the data as Avro files
   * @param schemaString The schema string of the data
   */
  def saveAsAvro[T <: GenericRecord : ClassTag](data: RDD[T], outputDir: String, schemaString: String): Unit = {
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
   * @param sc Spark context
   * @param path The path to a single Avro file (not the parent directory)
   * @param schemaString Optional schema string for reading
   * @tparam T The record type
   * @return List of records
   */
  def readFromSingleAvro[T <: GenericRecord : ClassTag](
      sc: SparkContext,
      path: String, schemaString: String = null): Seq[T] = {

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
   * @param sc Spark context
   * @param data The Avro data to write
   * @param path The path to a single Avro file (not the parent directory)
   * @param schemaString The schema string
   * @param forceOverwrite Optional parameter to force overwrite
   * @tparam T The record type
   */
  def saveAsSingleAvro[T <: GenericRecord : ClassTag](
      sc: SparkContext,
      data: Seq[T],
      path: String,
      schemaString: String,
      forceOverwrite: Boolean = false): Unit = {

    val fs = FileSystem.get(sc.hadoopConfiguration)
    val outputPath = new Path(path)
    val outputStream = fs.create(outputPath, forceOverwrite)
    val schema = new Parser().parse(schemaString)
    val writer = new GenericDatumWriter[T](schema)
    val dataFileWriter = new DataFileWriter[T](writer)
    dataFileWriter.create(schema, outputStream)
    data.foreach(datum => dataFileWriter.append(datum))
    dataFileWriter.close()
  }
}
