package com.linkedin.photon.ml.cli.game.scoring

import scala.collection.Map

import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.RDDLike
import com.linkedin.photon.ml.avro.AvroUtils
import com.linkedin.photon.ml.avro.data.{DataProcessingUtils, NameAndTermFeatureSetContainer, NameAndTerm}
import com.linkedin.photon.ml.avro.model.ModelProcessingUtils
import com.linkedin.photon.ml.contants.StorageLevel
import com.linkedin.photon.ml.data.GameData
import com.linkedin.photon.ml.evaluation.{RMSEEvaluator, BinaryClassificationEvaluator}
import com.linkedin.photon.ml.supervised.TaskType._
import com.linkedin.photon.ml.util._


/**
 * @author xazhang
 */
class Driver(val params: Params, val sparkContext: SparkContext, val logger: PhotonLogger) {

  import params._

  protected val parallelism: Int = sparkContext.getConf.get("spark.default.parallelism",
    s"${sparkContext.getExecutorStorageStatus.length * 3}").toInt
  protected val hadoopConfiguration = sparkContext.hadoopConfiguration

  protected val isAddingIntercept = true

  def prepareFeatureMaps(): Map[String, Map[NameAndTerm, Int]] = {

    val allFeatureSectionKeys = featureShardIdToFeatureSectionKeysMap.values.reduce(_ ++ _)
    val nameAndTermFeatureSetContainer = NameAndTermFeatureSetContainer.loadFromTextFiles(
      featureNameAndTermSetInputPath, allFeatureSectionKeys, hadoopConfiguration)

    val featureShardIdToFeatureMapMap =
      featureShardIdToFeatureSectionKeysMap.map { case (shardId, featureSectionKeys) =>
        val featureMap = nameAndTermFeatureSetContainer.getFeatureNameAndTermToIndexMap(featureSectionKeys,
          isAddingIntercept)
        (shardId, featureMap)
      }
    featureShardIdToFeatureMapMap.foreach { case (shardId, featureMap) =>
      logger.logDebug(s"Feature shard ID: $shardId, number of features: ${featureMap.size}")
    }
    featureShardIdToFeatureMapMap
  }


  def prepareGameDataSet(featureShardIdToFeatureMapMap: Map[String, Map[NameAndTerm, Int]]): RDD[(Long, GameData)] = {

    val recordsPath = dateRangeOpt match {
      case Some(dateRange) =>
        val Array(startDate, endDate) = dateRange.split("-")
        IOUtils.getInputPathsWithinDateRange(inputDirs, startDate, endDate, hadoopConfiguration, errorOnMissing = false)
      case None => inputDirs.toSeq
    }
    logger.logDebug(s"Avro records paths:\n${recordsPath.mkString("\n")}")
    val records = AvroUtils.readAvroFiles(sparkContext, recordsPath, parallelism)
    val globalDataPartitioner = new LongHashPartitioner(records.partitions.length)

    val gameDataSet = DataProcessingUtils.parseAndGenerateGameDataSetFromGenericRecords(records,
      featureShardIdToFeatureSectionKeysMap, featureShardIdToFeatureMapMap, randomEffectIdSet)
        .partitionBy(globalDataPartitioner)
        .setName("Scoring Game data set")
        .persist(StorageLevel.FREQUENT_REUSE_RDD_STORAGE_LEVEL)

    // Log some simple summary info on the Game data set
    logger.logDebug(s"Summary for the validating Game data set")
    val numSamples = gameDataSet.count()
    logger.logDebug(s"numSamples: $numSamples")
    val responseSum = gameDataSet.values.map(_.response).sum()
    logger.logDebug(s"responseSum: $responseSum")
    val weightSum = gameDataSet.values.map(_.weight).sum()
    logger.logDebug(s"weightSum: $weightSum")
    val randomEffectIdToIndividualIdMap = gameDataSet.values.first().randomEffectIdToIndividualIdMap
    randomEffectIdToIndividualIdMap.keySet.foreach { randomEffectId =>
      val dataStats = gameDataSet.values.map { gameData =>
        val individualId = gameData.randomEffectIdToIndividualIdMap(randomEffectId)
        (individualId, (gameData.response, 1))
      }.reduceByKey { case ((responseSum1, numSample1), (responseSum2, numSample2)) =>
        (responseSum1 + responseSum2, numSample1 + numSample2)
      }.cache()
      val responseSumStats = dataStats.values.map(_._1).stats()
      val numSamplesStats = dataStats.values.map(_._2).stats()
      logger.logDebug(s"numSamplesStats for $randomEffectId: $numSamplesStats")
      logger.logDebug(s"responseSumStats for $randomEffectId: $responseSumStats")
    }

    gameDataSet
  }


  def scoreAndWriteScoreToHDFS(
      featureShardIdToFeatureMapMap: Map[String, Map[NameAndTerm, Int]],
      gameDataSet: RDD[(Long, GameData)]): RDD[(Long, Double)] = {

    val gameModel = ModelProcessingUtils.loadGameModelFromHDFS(featureShardIdToFeatureMapMap, gameModelInputDir,
      sparkContext)

    gameModel.foreach {
      case rddLike: RDDLike => rddLike.persistRDD(StorageLevel.FREQUENT_REUSE_RDD_STORAGE_LEVEL)
      case _ =>
    }
    logger.logDebug(s"Loaded game model summary:\n${gameModel.map(_.toSummaryString).mkString("\n")}")

    val scores = gameModel.map(_.score(gameDataSet)).reduce(_ + _).scores
        .setName("Scores").persist(StorageLevel.FREQUENT_REUSE_RDD_STORAGE_LEVEL)
    val scoredItems = scores.join(gameDataSet).map { case (_, (score, gameData)) =>
      val ids = gameData.randomEffectIdToIndividualIdMap.values
      val label = gameData.response
      ScoredItem(ids, score, label)
    }.setName("Scored item").persist(StorageLevel.FREQUENT_REUSE_RDD_STORAGE_LEVEL)

    val numScoredItems = scoredItems.count()
    val scoresDir = new Path(outputDir, Driver.SCORES).toString
    Utils.deleteHDFSDir(scoresDir, hadoopConfiguration)
    // Should always materialize the scoredItems first (e.g., count()) before the coalesce happens
    scoredItems.coalesce(numFiles).saveAsTextFile(scoresDir)
    logger.logDebug(s"Number of scored items: $numScoredItems")

    scores
  }


  def evaluateAndLog(gameDataSet: RDD[(Long, GameData)], scores: RDD[(Long, Double)]): Unit = {

    val validatingLabelAndOffsets = gameDataSet.mapValues(gameData => (gameData.response, gameData.offset))
    val metric =  taskType match {
      case LOGISTIC_REGRESSION =>
        new BinaryClassificationEvaluator(validatingLabelAndOffsets).evaluate(scores)
      case LINEAR_REGRESSION =>
        val validatingLabelAndOffsetAndWeights = validatingLabelAndOffsets.mapValues { case (label, offset) =>
          (label, offset, 1.0)
        }
        new RMSEEvaluator(validatingLabelAndOffsetAndWeights).evaluate(scores)
      case _ =>
        throw new UnsupportedOperationException(s"Task type: $taskType is not supported to create validating " +
            s"evaluator")
    }
    logger.logInfo(s"Evaluation metric: $metric")
  }


  def run(): Unit = {

    var startTime = System.nanoTime()
    val featureShardIdToFeatureMapMap = prepareFeatureMaps()
    val initializationTime = (System.nanoTime() - startTime) * 1e-9
    logger.logInfo(s"Time elapsed after preparing feature maps: $initializationTime (s)\n")
    logger.flush()

    startTime = System.nanoTime()
    val gameDataSet = prepareGameDataSet(featureShardIdToFeatureMapMap)
    val gameDataSetPreparationTime = (System.nanoTime() - startTime) * 1e-9
    logger.logInfo(s"Time elapsed after game data set preparation: $gameDataSetPreparationTime (s)\n")
    logger.flush()

    startTime = System.nanoTime()
    val scores = scoreAndWriteScoreToHDFS(featureShardIdToFeatureMapMap, gameDataSet)
    val scoringTime = (System.nanoTime() - startTime) * 1e-9
    logger.logInfo(s"Time elapsed scoring and writing scores to HDFS: $scoringTime (s)\n")
    logger.flush()

    startTime = System.nanoTime()
    evaluateAndLog(gameDataSet, scores)
    val postprocessingTime = (System.nanoTime() - startTime) * 1e-9
    logger.logInfo(s"Time elapsed after evaluation: $postprocessingTime (s)\n")
    logger.flush()
  }
}

object Driver {
  private val SCORES = "scores"
  private val LOGS = "logs"

  def main(args: Array[String]): Unit = {

    val startTime = System.nanoTime()

    val params = Params.parseFromCommandLine(args)
    import params._

    val sc = SparkContextConfiguration.asYarnClient(applicationName)

    val logsDir = new Path(outputDir, LOGS).toString
    Utils.createHDFSDir(logsDir, sc.hadoopConfiguration)
    val logger = new PhotonLogger(logsDir, sc.hadoopConfiguration)
    logger.logDebug(params.toString + "\n")
    logger.flush()
    val job = new Driver(params, sc, logger)
    job.run()

    val timeElapsed = (System.nanoTime() - startTime) * 1e-9 / 60
    logger.logInfo(s"Overall time elapsed $timeElapsed minutes")

    logger.close()
    sc.stop()
  }
}
