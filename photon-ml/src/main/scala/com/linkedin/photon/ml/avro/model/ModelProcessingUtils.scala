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

import scala.collection.Map

import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.avro.generated.BayesianLinearModelAvro
import com.linkedin.photon.ml.avro.{AvroIOUtils, AvroUtils}
import com.linkedin.photon.ml.avro.data.NameAndTerm
import com.linkedin.photon.ml.model.{Coefficients, RandomEffectModel, FixedEffectModel, Model}
import com.linkedin.photon.ml.util.{IOUtils, Utils}


/**
 * Some basic functions to read/write GAME models from/to HDFS. The current implementaion assumes the models are stored
 * using Avro format.
 * @author xazhang
 */
//TODO: Change the scope of all functions in the object to [[com.linkedin.photon.ml.avro]] after Avro related classes/functons are decoupled from the rest of code
object ModelProcessingUtils {
  private val DEFAULT_AVRO_FILE_NAME = "part-00000.avro"
  private val ID_INFO = "id-info"
  private val COEFFICIENTS = "coefficients"
  private val FIXED_EFFECT = "fixed-effect"
  private val RANDOM_EFFECT = "random-effect"

  protected[ml] def saveGameModelsToHDFS(
      gameModel: Iterable[Model],
      featureShardIdToFeatureMapMap: Map[String, Map[NameAndTerm, Int]],
      outputDir: String,
      sparkContext: SparkContext): Unit = {

    val configuration = sparkContext.hadoopConfiguration
    val featureShardIdToFeatureSwappedMapBroadcastMap = featureShardIdToFeatureMapMap.map { case (shardId, map) =>
      (shardId, sparkContext.broadcast(map.map(_.swap)))
    }
    gameModel.foreach { case model =>
      model match {
        case fixedEffectModel: FixedEffectModel =>
          val featureShardId = fixedEffectModel.featureShardId
          val fixedEffectModelOutputDir = new Path(outputDir, s"$FIXED_EFFECT/$featureShardId").toString
          Utils.createHDFSDir(fixedEffectModelOutputDir, configuration)

          //Write the model ID info
          val modelIdInfoPath = new Path(fixedEffectModelOutputDir, ID_INFO)
          val id = Array(featureShardId)
          IOUtils.writeStringsToHDFS(id.iterator, modelIdInfoPath, configuration, forceOverwrite = false)

          //Write the coefficients
          val coefficientsOutputDir = new Path(fixedEffectModelOutputDir, COEFFICIENTS).toString
          Utils.createHDFSDir(coefficientsOutputDir, configuration)
          val featureIndexToNameAndTermMap = featureShardIdToFeatureSwappedMapBroadcastMap(featureShardId).value
          val coefficients = fixedEffectModel.coefficients
          saveCoefficientsToHDFS(coefficients, featureIndexToNameAndTermMap, coefficientsOutputDir, sparkContext)

        case randomEffectModel: RandomEffectModel =>
          val randomEffectId = randomEffectModel.randomEffectId
          val featureShardId = randomEffectModel.featureShardId

          val randomEffectModelOutputDir = new Path(outputDir, s"$RANDOM_EFFECT/$randomEffectId-$featureShardId")
          Utils.createHDFSDir(randomEffectModelOutputDir.toString, configuration)

          //Write the model ID info
          val modelIdInfoPath = new Path(randomEffectModelOutputDir, ID_INFO)
          val ids = Array(randomEffectId, featureShardId)
          IOUtils.writeStringsToHDFS(ids.iterator, modelIdInfoPath, configuration, forceOverwrite = false)

          //Write the coefficientsRDD
          val coefficientsRDDOutputDir = new Path(randomEffectModelOutputDir, COEFFICIENTS).toString
          val featureIndexToNameAndTermMapBroadcast = featureShardIdToFeatureSwappedMapBroadcastMap(featureShardId)
          val coefficientsRDD = randomEffectModel.coefficientsRDD
          saveCoefficientsRDDToHDFS(coefficientsRDD, featureIndexToNameAndTermMapBroadcast, coefficientsRDDOutputDir)
      }
    }
  }

  protected[ml] def loadGameModelFromHDFS(
      featureShardIdToFeatureMapMap: Map[String, Map[NameAndTerm, Int]],
      inputDir: String,
      sparkContext: SparkContext): Iterable[Model] = {

    val configuration = sparkContext.hadoopConfiguration
    val inputDirAsPath = new Path(inputDir)
    val fs = inputDirAsPath.getFileSystem(configuration)

    val featureShardIdToFeatureMapBroadcastMap = featureShardIdToFeatureMapMap
        .map { case (featureShardId, featureMap) => (featureShardId, sparkContext.broadcast(featureMap)) }

    // Load the fixed effect models
    val fixedEffectModelInputDir = new Path(inputDir, FIXED_EFFECT)
    val fixedEffectModels = if (fs.exists(fixedEffectModelInputDir)) {
      fs.listStatus(fixedEffectModelInputDir).map { fileStatus =>
        val inputPath = fileStatus.getPath

        // Load the model ID info
        val idInfoPath = new Path(inputPath, ID_INFO)
        val Array(featureShardId) = IOUtils.readStringsFromHDFS(idInfoPath, configuration).toArray

        // Load the coefficients
        val featureNameAndTermToIndexMap = featureShardIdToFeatureMapMap(featureShardId)
        val modelPath = new Path(inputPath, COEFFICIENTS)
        val coefficients = loadCoefficientsFromHDFS(modelPath.toString, featureNameAndTermToIndexMap, sparkContext)
        new FixedEffectModel(sparkContext.broadcast(coefficients), featureShardId)
      }
    } else {
      Array[FixedEffectModel]()
    }

    // Load the random effect models
    val randomEffectModelInputDir = new Path(inputDir, RANDOM_EFFECT)
    val randomEffectModels = if (fs.exists(randomEffectModelInputDir)) {
      fs.listStatus(randomEffectModelInputDir).map { innerFileStatus =>
        val innerPath = innerFileStatus.getPath

        // Load the model ID info
        val idInfoPath = new Path(innerPath, ID_INFO)
        val Array(randomEffectId, featureShardId) = IOUtils.readStringsFromHDFS(idInfoPath, configuration).toArray

        // Load the coefficients
        val featureNameAndTermToIndexMap = featureShardIdToFeatureMapBroadcastMap(featureShardId)
        val coefficientsRDDInputPath = new Path(innerPath, COEFFICIENTS)
        val coefficientsRDD = loadCoefficientsRDDFromHDFS(coefficientsRDDInputPath.toString,
          featureNameAndTermToIndexMap, sparkContext)
        new RandomEffectModel(coefficientsRDD, randomEffectId, featureShardId)
      }
    } else {
      Array[RandomEffectModel]()
    }

    fixedEffectModels ++ randomEffectModels
  }

  private def saveCoefficientsToHDFS(
      coefficients: Coefficients,
      featureIndexToNameAndTermMap: Map[Int, NameAndTerm],
      outputDir: String,
      sparkContext: SparkContext): Unit = {

    val bayesianLinearModelAvro = AvroUtils.modelToBayesianLinearModelAvro(coefficients, FIXED_EFFECT,
      featureIndexToNameAndTermMap)
    val coefficientsOutputPath = new Path(outputDir, DEFAULT_AVRO_FILE_NAME).toString
    AvroIOUtils.saveAsSingleAvro(sparkContext, Seq(bayesianLinearModelAvro), coefficientsOutputPath,
      BayesianLinearModelAvro.getClassSchema.toString, forceOverwrite = false)
  }

  //TODO: Currently only the means of the coefficients are loaded, the variances are discarded
  private def loadCoefficientsFromHDFS(
      inputDir: String,
      featureNameAndTermToIndexMap: Map[NameAndTerm, Int],
      sparkContext: SparkContext): Coefficients = {

    val coefficientsPath = new Path(inputDir, DEFAULT_AVRO_FILE_NAME).toString
    val linearModelAvroSchema = BayesianLinearModelAvro.getClassSchema.toString
    val linearModelAvro = AvroIOUtils.readFromSingleAvro[BayesianLinearModelAvro](sparkContext, coefficientsPath,
      linearModelAvroSchema).head
    val means = AvroUtils.loadMeanVectorFromBayesianLinearModelAvro(linearModelAvro, featureNameAndTermToIndexMap)
    Coefficients(means, variancesOption = None)
  }

  private def saveCoefficientsRDDToHDFS(
      coefficientsRDD: RDD[(String, Coefficients)],
      featureIndexToNameAndTermMapBroadcast: Broadcast[Map[Int, NameAndTerm]],
      outputDir: String): Unit = {

    val linearModelAvro = coefficientsRDD.map { case (modelId, coefficients) =>
      AvroUtils.modelToBayesianLinearModelAvro(coefficients, modelId, featureIndexToNameAndTermMapBroadcast.value)
    }
    AvroIOUtils.saveAsAvro(linearModelAvro, outputDir, BayesianLinearModelAvro.getClassSchema.toString)
  }

  //TODO: Currently only the means of the coefficients are loaded, the variances are discarded
  private def loadCoefficientsRDDFromHDFS(
      coefficientsRDDInputDir: String,
      featureIndexToNameAndTermMapBroadcast: Broadcast[Map[NameAndTerm, Int]],
      sparkContext: SparkContext): RDD[(String, Coefficients)] = {

    val modelAvros = AvroIOUtils.readFromAvro[BayesianLinearModelAvro](sparkContext, coefficientsRDDInputDir,
      minNumPartitions = sparkContext.defaultParallelism)
    modelAvros.map { modelAvro =>
      val modelId = modelAvro.getModelId.toString
      val nameAndTermFeatureMap = featureIndexToNameAndTermMapBroadcast.value
      val means = AvroUtils.loadMeanVectorFromBayesianLinearModelAvro(modelAvro, nameAndTermFeatureMap)
      (modelId, Coefficients(means, variancesOption = None))
    }
  }

  private def combineCoefficients(coefficients1: Coefficients, coefficients2: Coefficients): Coefficients = {
    val combinedMeans = coefficients1.means + coefficients2.means
    Coefficients(combinedMeans, variancesOption = None)
  }

  protected[ml] def collapseGameModel(gameModel: Map[String, Model], sparkContext: SparkContext)
  : Map[(String, String), Model] = {

    gameModel.toArray.map {
      case (modelId, fixedEffectModel: FixedEffectModel) =>
        val effectId = FIXED_EFFECT
        val featureShardId = fixedEffectModel.featureShardId
        Pair[Pair[String, String], Model]((effectId, featureShardId), fixedEffectModel)
      case (modelId, randomEffectModel: RandomEffectModel) =>
        val effectId = randomEffectModel.randomEffectId
        val featureShardId = randomEffectModel.featureShardId
        Pair[Pair[String, String], Model]((effectId, featureShardId), randomEffectModel)
      case (modelId, model) =>
        throw new UnsupportedOperationException(s"Collapse model of type ${model.getClass} is not supported")
    }
        .groupBy(_._1)
        .map { case ((effectId, featureShardId), modelsMap) =>
      val combinedModel = modelsMap.map(_._2).reduce[Model] {
        case (fixedEffectModel1: FixedEffectModel, fixedEffectModel2: FixedEffectModel) =>
          val combinedCoefficients = combineCoefficients(fixedEffectModel1.coefficients, fixedEffectModel2.coefficients)
          val combinedCoefficientsBroadcast = sparkContext.broadcast(combinedCoefficients)
          new FixedEffectModel(combinedCoefficientsBroadcast, featureShardId)

        case (randomEffectModel1: RandomEffectModel, randomEffectModel2: RandomEffectModel) =>
          val coefficientsRDD1 = randomEffectModel1.coefficientsRDD
          val coefficientsRDD2 = randomEffectModel2.coefficientsRDD
          val combinedCoefficientsRDD = coefficientsRDD1.cogroup(coefficientsRDD2)
              .mapValues { case (coefficientsItr1, coefficientsItr2) =>
            assert(coefficientsItr1.size == 1)
            assert(coefficientsItr2.size == 1)
            combineCoefficients(coefficientsItr1.head, coefficientsItr2.head)
          }
          new RandomEffectModel(combinedCoefficientsRDD, effectId, featureShardId)

        case (model1, model2) =>
          throw new UnsupportedOperationException(s"Combining models of type ${model1.getClass} " +
              s"and ${model2.getClass} is not supported!")
      }

      ((effectId, featureShardId), combinedModel)
    }
  }
}
