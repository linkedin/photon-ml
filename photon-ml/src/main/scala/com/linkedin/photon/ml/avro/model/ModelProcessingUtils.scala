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
package com.linkedin.photon.ml.avro.model

import breeze.linalg.Vector
import com.linkedin.photon.ml.avro.{AvroIOUtils, AvroUtils}
import com.linkedin.photon.ml.avro.data.NameAndTerm
import com.linkedin.photon.ml.avro.generated.{BayesianLinearModelAvro, LatentFactorAvro}
import com.linkedin.photon.ml.model._
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import com.linkedin.photon.ml.supervised.regression.LinearRegressionModel
import com.linkedin.photon.ml.util.{IOUtils, Utils}
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD

import scala.collection.Map
import scala.reflect._

/**
  * Some basic functions to read/write GAME models from/to HDFS. The current implementation assumes the models are stored
  * using Avro format.
  */
object ModelProcessingUtils {

  import com.linkedin.photon.ml.avro.Constants._

  // TODO: Change the scope of all functions in the object to [[com.linkedin.photon.ml.avro]] after Avro related
  // classes/functions are decoupled from the rest of code
  protected[ml] def saveGameModelsToHDFS(
      gameModel: GAMEModel,
      featureShardIdToFeatureMapMap: Map[String, Map[NameAndTerm, Int]],
      outputDir: String,
      numberOfOutputFilesForRandomEffectModel: Int,
      sparkContext: SparkContext): Unit = {

    val configuration = sparkContext.hadoopConfiguration
    val featureShardIdToFeatureSwappedMapBroadcastMap = featureShardIdToFeatureMapMap.map { case (shardId, map) =>
      (shardId, sparkContext.broadcast(map.map(_.swap)))
    }
    gameModel.toMap.foreach { case (name, model) =>
      model match {
        case fixedEffectModel: FixedEffectModel =>
          val featureShardId = fixedEffectModel.featureShardId
          val fixedEffectModelOutputDir = new Path(outputDir, s"$FIXED_EFFECT/$name").toString
          Utils.createHDFSDir(fixedEffectModelOutputDir, configuration)

          //Write the model ID info
          val modelIdInfoPath = new Path(fixedEffectModelOutputDir, ID_INFO)
          val id = Array(featureShardId)
          IOUtils.writeStringsToHDFS(id.iterator, modelIdInfoPath, configuration, forceOverwrite = false)

          //Write the coefficients
          val coefficientsOutputDir = new Path(fixedEffectModelOutputDir, COEFFICIENTS).toString
          Utils.createHDFSDir(coefficientsOutputDir, configuration)
          val featureIndexToNameAndTermMap = featureShardIdToFeatureSwappedMapBroadcastMap(featureShardId).value
          val model = fixedEffectModel.model
          saveModelToHDFS(model, featureIndexToNameAndTermMap, coefficientsOutputDir, sparkContext)

        case randomEffectModel: RandomEffectModel =>
          val randomEffectId = randomEffectModel.randomEffectId
          val featureShardId = randomEffectModel.featureShardId

          val randomEffectModelOutputDir = new Path(outputDir, s"$RANDOM_EFFECT/$name")
          //Write the model ID info
          val modelIdInfoPath = new Path(randomEffectModelOutputDir, ID_INFO)
          val ids = Array(randomEffectId, featureShardId)
          IOUtils.writeStringsToHDFS(ids.iterator, modelIdInfoPath, configuration, forceOverwrite = false)
          val featureIndexToNameAndTermMapBroadcast = featureShardIdToFeatureSwappedMapBroadcastMap(featureShardId)
          saveRandomEffectModelToHDFS(randomEffectModel, featureIndexToNameAndTermMapBroadcast,
            randomEffectModelOutputDir, numberOfOutputFilesForRandomEffectModel, configuration)
      }
    }
  }

  protected[ml] def loadGameModelFromHDFS[GLM <: GeneralizedLinearModel : ClassTag](
      featureShardIdToFeatureMapMap: Map[String, Map[NameAndTerm, Int]],
      inputDir: String,
      sparkContext: SparkContext): GAMEModel = {

    val configuration = sparkContext.hadoopConfiguration
    val inputDirAsPath = new Path(inputDir)
    val fs = inputDirAsPath.getFileSystem(configuration)

    val featureShardIdToFeatureMapBroadcastMap = featureShardIdToFeatureMapMap.map {
      case (featureShardId, featureMap) => (featureShardId, sparkContext.broadcast(featureMap))
    }

    // Load the fixed effect models
    val fixedEffectModelInputDir = new Path(inputDir, FIXED_EFFECT)
    val fixedEffectModels = if (fs.exists(fixedEffectModelInputDir)) {
      fs.listStatus(fixedEffectModelInputDir).map { fileStatus =>
        val innerPath = fileStatus.getPath
        val name = innerPath.getName

        // Load the model ID info
        val idInfoPath = new Path(innerPath, ID_INFO)
        val Array(featureShardId) = IOUtils.readStringsFromHDFS(idInfoPath, configuration).toArray

        // Load the coefficients
        val featureNameAndTermToIndexMap = featureShardIdToFeatureMapMap(featureShardId)
        val modelPath = new Path(innerPath, COEFFICIENTS)
        val coefficients = loadCoefficientsFromHDFS(modelPath.toString, featureNameAndTermToIndexMap, sparkContext)
        val glm = GeneralizedLinearModel.fromCoefficients[GLM](coefficients)

        (name, new FixedEffectModel(sparkContext.broadcast(glm), featureShardId))
      }
    } else {
      Array[(String, FixedEffectModel)]()
    }

    // Load the random effect models
    val randomEffectModelInputDir = new Path(inputDir, RANDOM_EFFECT)
    val randomEffectModels = if (fs.exists(randomEffectModelInputDir)) {
      fs.listStatus(randomEffectModelInputDir).map { innerFileStatus =>
        val innerPath = innerFileStatus.getPath
        val name = innerPath.getName

        // Load the model ID info
        val idInfoPath = new Path(innerPath, ID_INFO)
        val Array(randomEffectId, featureShardId) = IOUtils.readStringsFromHDFS(idInfoPath, configuration).toArray

        // Load the models
        val featureNameAndTermToIndexMap = featureShardIdToFeatureMapBroadcastMap(featureShardId)
        val modelsRDDInputPath = new Path(innerPath, COEFFICIENTS)
        val modelsRDD = loadModelsRDDFromHDFS[GLM](modelsRDDInputPath.toString, featureNameAndTermToIndexMap, sparkContext)

        (name, new RandomEffectModel(modelsRDD, randomEffectId, featureShardId))
      }
    } else {
      Array[(String, RandomEffectModel)]()
    }

    val gameModels = fixedEffectModels ++ randomEffectModels
    val gameModelNames = gameModels.map(_._1)

    require(gameModelNames.toSet.size == gameModelNames.length,
      s"Duplicated model names found: ${gameModelNames.mkString("\t")}")

    new GAMEModel(gameModels.toMap)
  }

  private def saveRandomEffectModelToHDFS(
      randomEffectModel: RandomEffectModel,
      featureIndexToNameAndTermMapBroadcast: Broadcast[Map[Int, NameAndTerm]],
      randomEffectModelOutputDir: Path,
      numberOfOutputFilesForRandomEffectModel: Int,
      configuration: Configuration): Unit = {

    Utils.createHDFSDir(randomEffectModelOutputDir.toString, configuration)

    //Write the coefficientsRDD
    val coefficientsRDDOutputDir = new Path(randomEffectModelOutputDir, COEFFICIENTS).toString
    val modelsRDD = if (numberOfOutputFilesForRandomEffectModel > 0){
        // Control the number of output files by re-partitioning the RDD.
        randomEffectModel.modelsRDD.coalesce(numberOfOutputFilesForRandomEffectModel)
      } else {
        randomEffectModel.modelsRDD
      }

    saveModelsRDDToHDFS(modelsRDD, featureIndexToNameAndTermMapBroadcast, coefficientsRDDOutputDir)
  }

  private def saveModelToHDFS(
      model: GeneralizedLinearModel,
      featureIndexToNameAndTermMap: Map[Int, NameAndTerm],
      outputDir: String,
      sparkContext: SparkContext): Unit = {

    val bayesianLinearModelAvro = AvroUtils.convertGLMModelToBayesianLinearModelAvro(
      model,
      FIXED_EFFECT,
      featureIndexToNameAndTermMap)
    val modelOutputPath = new Path(outputDir, DEFAULT_AVRO_FILE_NAME).toString
    AvroIOUtils.saveAsSingleAvro(
      sparkContext,
      Seq(bayesianLinearModelAvro),
      modelOutputPath,
      BayesianLinearModelAvro.getClassSchema.toString,
      forceOverwrite = false)
  }

  // TODO: Currently only the means of the coefficients are loaded, the variances are discarded
  // TODO: Currently the model type is not loaded; all models are treated as LinearRegressionModels
  private def loadCoefficientsFromHDFS(
      inputDir: String,
      featureNameAndTermToIndexMap: Map[NameAndTerm, Int],
      sparkContext: SparkContext): Coefficients = {

    val coefficientsPath = new Path(inputDir, DEFAULT_AVRO_FILE_NAME).toString
    val linearModelAvroSchema = BayesianLinearModelAvro.getClassSchema.toString
    val linearModelAvro = AvroIOUtils.readFromSingleAvro[BayesianLinearModelAvro](sparkContext, coefficientsPath,
      linearModelAvroSchema).head
    val means = AvroUtils.convertBayesianLinearModelAvroToMeanVector(linearModelAvro, featureNameAndTermToIndexMap)
    Coefficients(means, variancesOption = None)
  }

  private def saveModelsRDDToHDFS(
      modelsRDD: RDD[(String, GeneralizedLinearModel)],
      featureIndexToNameAndTermMapBroadcast: Broadcast[Map[Int, NameAndTerm]],
      outputDir: String): Unit = {

    val linearModelAvro = modelsRDD.map { case (modelId, model) =>
      AvroUtils.convertGLMModelToBayesianLinearModelAvro(model, modelId, featureIndexToNameAndTermMapBroadcast.value)
    }

    AvroIOUtils.saveAsAvro(linearModelAvro, outputDir, BayesianLinearModelAvro.getClassSchema.toString)
  }

  // TODO: Currently only the means of the coefficients are loaded, the variances are discarded
  // TODO: Currently the model type is not loaded; all models are treated as LinearRegressionModels
  private def loadModelsRDDFromHDFS[GLM <: GeneralizedLinearModel : ClassTag](
      coefficientsRDDInputDir: String,
      featureIndexToNameAndTermMapBroadcast: Broadcast[Map[NameAndTerm, Int]],
      sparkContext: SparkContext): RDD[(String, GeneralizedLinearModel)] = {

    val modelAvros = AvroIOUtils.readFromAvro[BayesianLinearModelAvro](
      sparkContext,
      coefficientsRDDInputDir,
      minNumPartitions = sparkContext.defaultParallelism)
    modelAvros.map { modelAvro =>
      val modelId = modelAvro.getModelId.toString
      val nameAndTermFeatureMap = featureIndexToNameAndTermMapBroadcast.value
      val means = AvroUtils.convertBayesianLinearModelAvroToMeanVector(modelAvro, nameAndTermFeatureMap)
      val glm = GeneralizedLinearModel.fromCoefficients(Coefficients(means))

      (modelId, glm)
    }
  }

  /**
    * Save the matrix factorization model of type [[MatrixFactorizationModel]] to HDFS as Avro files
    *
    * @param matrixFactorizationModel The given matrix factorization model
    * @param outputDir The HDFS output directory for the matrix factorization model
    * @param numOutputFiles Number of output files to generate for row/column latent factors of the matrix
    *                       factorization model
    * @param sparkContext The Spark context
    */
  protected[ml] def saveMatrixFactorizationModelToHDFS(
      matrixFactorizationModel: MatrixFactorizationModel,
      outputDir: String,
      numOutputFiles: Int,
      sparkContext: SparkContext): Unit = {

    val rowLatentFactors = matrixFactorizationModel.rowLatentFactors
    val rowEffectType = matrixFactorizationModel.rowEffectType
    val rowLatentFactorsOutputDir = new Path(outputDir, rowEffectType).toString
    val rowLatentFactorsAvro = rowLatentFactors.coalesce(numOutputFiles).map { case (rowId, latentFactor) =>
      AvroUtils.convertLatentFactorToLatentFactorAvro(rowId, latentFactor)
    }
    AvroIOUtils.saveAsAvro(rowLatentFactorsAvro, rowLatentFactorsOutputDir, LatentFactorAvro.getClassSchema.toString)

    val colLatentFactors = matrixFactorizationModel.colLatentFactors
    val colEffectType = matrixFactorizationModel.colEffectType
    val colLatentFactorsOutputDir = new Path(outputDir, colEffectType).toString
    val colLatentFactorsAvro = colLatentFactors.coalesce(numOutputFiles).map { case (colId, latentFactor) =>
      AvroUtils.convertLatentFactorToLatentFactorAvro(colId, latentFactor)
    }
    AvroIOUtils.saveAsAvro(colLatentFactorsAvro, colLatentFactorsOutputDir, LatentFactorAvro.getClassSchema.toString)
  }

  private def loadLatentFactorsFromHDFS(inputDir: String, sparkContext: SparkContext): RDD[(String, Vector[Double])] = {
    val minNumPartitions = sparkContext.defaultParallelism
    val modelAvros = AvroIOUtils.readFromAvro[LatentFactorAvro](sparkContext, inputDir, minNumPartitions)
    modelAvros.map(AvroUtils.convertLatentFactorAvroToLatentFactor)
  }

  /**
    * Load the matrix factorization model of type [[MatrixFactorizationModel]] from the Avro files on HDFS
    *
    * @param inputDir The input directory of the Avro files on HDFS
    * @param rowEffectType What each row of the matrix corresponds to, e.g., memberId or itemId
    * @param colEffectType What each column of the matrix corresponds to, e.g., memberId or itemId
    * @param sparkContext The Spark context
    * @return The loaded matrix factorization model of type [[MatrixFactorizationModel]]
    */
  protected[ml] def loadMatrixFactorizationModelFromHDFS(
      inputDir: String,
      rowEffectType: String,
      colEffectType: String,
      sparkContext: SparkContext): MatrixFactorizationModel = {

    val configuration = sparkContext.hadoopConfiguration
    val inputDirAsPath = new Path(inputDir)
    val fs = inputDirAsPath.getFileSystem(configuration)
    assert(fs.exists(inputDirAsPath),
      s"Specified input directory $inputDir for matrix factorization model is not found!")
    val rowLatentFactorsPath = new Path(inputDir, rowEffectType)
    assert(fs.exists(rowLatentFactorsPath),
      s"Specified input directory $rowLatentFactorsPath for row latent factors is not found!")
    val rowLatentFactors = loadLatentFactorsFromHDFS(rowLatentFactorsPath.toString, sparkContext)
    val colLatentFactorsPath = new Path(inputDir, colEffectType)
    assert(fs.exists(colLatentFactorsPath),
      s"Specified input directory $colLatentFactorsPath for column latent factors is not found!")
    val colLatentFactors = loadLatentFactorsFromHDFS(colLatentFactorsPath.toString, sparkContext)
    new MatrixFactorizationModel(rowEffectType, colEffectType, rowLatentFactors, colLatentFactors)
  }
}
