package com.linkedin.photon.ml.avro.kddcup2012

import java.io.{InputStreamReader, BufferedReader}

import scala.collection.mutable
import scala.collection.Map
import scala.collection.JavaConversions._
import scala.collection.mutable.ArrayBuffer

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.apache.spark.{SparkContext, SparkConf}

import com.linkedin.photon.ml.avro.generated.{KDDCup2012DataAvro, NameTermValueAvro}
import com.linkedin.photon.ml.io.AvroIOUtils
import com.linkedin.photon.ml.util.Utils


/**
 * @author xazhang
 */
object GenerateGameData {

  def readIdTokensMapFromPath(path: Path, configuration: Configuration): Map[Int, Array[Int]] = {
    val fs = path.getFileSystem(configuration)
    val bufferedReader = new BufferedReader(new InputStreamReader(fs.open(path)))
    val idTokensBuffer = new mutable.ArrayBuffer[(Int, Array[Int])]
    var line = bufferedReader.readLine()
    while (line != null) {
      val Array(rawId, tokensAsOneString) = line.split("\t")
      val id = rawId.toInt
      val tokens = tokensAsOneString.split("\\|").map(_.toInt)
      idTokensBuffer += ((id, tokens))
      line = bufferedReader.readLine()
    }
    bufferedReader.close()
    idTokensBuffer.toMap
  }

  case class UserProfileFeatures(gender: Int, age: Int)

  def readUserProfileFeaturesFromPath(path: Path, configuration: Configuration): Map[Int, UserProfileFeatures] = {
    val fs = path.getFileSystem(configuration)
    val bufferedReader = new BufferedReader(new InputStreamReader(fs.open(path)))
    val userIdToProfileFeaturesBuffer = new mutable.ArrayBuffer[(Int, UserProfileFeatures)]
    var line = bufferedReader.readLine()
    while (line != null) {
      val Array(rawId, rawGender, rawAge) = line.split("\t")
      val id = rawId.toInt
      val gender = rawGender.toInt
      val age = rawAge.toInt
      userIdToProfileFeaturesBuffer += ((id, UserProfileFeatures(gender, age)))
      line = bufferedReader.readLine()
    }
    bufferedReader.close()
    userIdToProfileFeaturesBuffer.toMap
  }

  def main(args: Array[String]): Unit = {

    val sparkConf = new SparkConf().setAppName("GenerateGameData")
    val sparkContext = new SparkContext(sparkConf)
    val hadoopConfiguration = sparkContext.hadoopConfiguration

    assert(args.length == 2, "Expected args consists of two parts, the input directory and the output directory")
    val inputDir = args(0)
    val outputDir = args(1)
    Utils.createHDFSDir(outputDir.toString, hadoopConfiguration)

    val trainingDataInputPath = new Path(inputDir, "training.txt")

    val publicTestingDataInputPath = new Path(inputDir, "publicTestingData.txt")
    val privateTestingDataInputPath = new Path(inputDir, "privateTestingData.txt")

    val trainingDataOutputPath = new Path(outputDir, "training")
    val publicTestingDataOutputPath = new Path(outputDir, "publicTesting")
    val privateTestingDataOutputPath = new Path(outputDir, "privateTesting")

    // Get the queryIdTokensMap
    val queryIdToTokensIdPath = new Path(inputDir, "queryid_tokensid.txt")
    val queryIdToTokensMap = readIdTokensMapFromPath(queryIdToTokensIdPath, hadoopConfiguration)
    val queryIdToTokensMapBroadcast = sparkContext.broadcast(queryIdToTokensMap)
    println("queryIdToTokensMap is ready")

    // Get the purchasedKeywordIdTokensIdMap
    val purchasedKeywordIdTokensIdPath = new Path(inputDir, "purchasedkeywordid_tokensid.txt")
    val purchasedKeywordIdTokensIdMap = readIdTokensMapFromPath(purchasedKeywordIdTokensIdPath, hadoopConfiguration)
    val purchasedKeywordIdTokensIdMapBroadcast = sparkContext.broadcast(purchasedKeywordIdTokensIdMap)
    println("purchasedKeywordIdTokensIdMap is ready")

    // Get the titleIdTokensIdMap
    val titleIdTokensIdPath = new Path(inputDir, "titleid_tokensid.txt")
    val titleIdTokensIdMap = readIdTokensMapFromPath(titleIdTokensIdPath, hadoopConfiguration)
    val titleIdTokensIdMapBroadcast = sparkContext.broadcast(titleIdTokensIdMap)
    println("titleIdTokensIdMap is ready")

    // Get the descriptionIdTokensIdMap
    val descriptionIdTokensIdPath = new Path(inputDir, "descriptionid_tokensid.txt")
    val descriptionIdTokensIdMap = readIdTokensMapFromPath(descriptionIdTokensIdPath, hadoopConfiguration)
    val descriptionIdTokensIdMapBroadcast = sparkContext.broadcast(descriptionIdTokensIdMap)
    println("descriptionIdTokensIdMap is ready")

    // Get the userIdProfileFeatures
    val userIdProfilePath = new Path(inputDir, "userid_profile.txt")
    val userIdProfileFeaturesMap = readUserProfileFeaturesFromPath(userIdProfilePath, hadoopConfiguration)
    val userIdProfileFeaturesMapBroadcast = sparkContext.broadcast(userIdProfileFeaturesMap)
    println("userIdProfileFeaturesMap is ready")

    // Join the data with the features
    def joinDataWithFeaturesAndWriteAsAvroToHDFS(
        inputPath: Path, outputPath: Path, minPartitions: Int, isTraining: Boolean): Unit = {

      val records = sparkContext.textFile(inputPath.toString, minPartitions).flatMap { line =>
        val tokens = line.split("\t")
        var click = tokens(0).toInt
        var impression = tokens(1).toInt
        val adId = tokens(3).toInt
        val advertiserId = tokens(4).toInt
        val depth = tokens(5)
        val position = tokens(6)
        val queryId = tokens(7).toInt
        val keywordId = tokens(8).toInt
        val titleId = tokens(9).toInt
        val descriptionId = tokens(10).toInt
        val userId = tokens(11).toInt

        val queryTokens = queryIdToTokensMapBroadcast.value.getOrElse(queryId, Array(-1))
        val purchasedKeywordTokens = purchasedKeywordIdTokensIdMapBroadcast.value.getOrElse(keywordId, Array(-1))
        val titleTokens = titleIdTokensIdMapBroadcast.value.getOrElse(titleId, Array(-1))
        val descriptionTokens = descriptionIdTokensIdMapBroadcast.value.getOrElse(descriptionId, Array(-1))
        val userProfileFeatures = userIdProfileFeaturesMapBroadcast.value.getOrElse(userId, UserProfileFeatures(-1, -1))

        // Generate globalFeatures
        val positionFeature = NameTermValueAvro.newBuilder().setName("PositionId").setTerm(position).setValue(1).build()
        val depthFeature = NameTermValueAvro.newBuilder().setName("Depth").setTerm(depth).setValue(1).build()
        val positionWithDepthFeature = NameTermValueAvro.newBuilder().setName("PositionId|Depth")
            .setTerm(position + "|" + depth).setValue(1).build()
        val globalFeaturesAvro = List(positionFeature, depthFeature, positionWithDepthFeature)

        // Generate queryTokensFeatures
        val queryTokensFeaturesAvro = queryTokens.map(token =>
          NameTermValueAvro.newBuilder().setName("queryToken").setTerm(token.toString).setValue(1).build()
        ).toList

        // Generate purchasedKeywordTokensFeatures
        val purchasedKeywordTokensFeaturesAvro = purchasedKeywordTokens.map(token =>
          NameTermValueAvro.newBuilder().setName("purchasedKeywordToken").setTerm(token.toString).setValue(1).build()
        ).toList

        // Generate titleTokensFeatures
        val titleTokensFeaturesAvro = titleTokens.map(token =>
          NameTermValueAvro.newBuilder().setName("titleToken").setTerm(token.toString).setValue(1).build()
        ).toList

        // Generate descriptionTokensFeatures
        val descriptionTokensFeaturesAvro = descriptionTokens.map(token =>
          NameTermValueAvro.newBuilder().setName("descriptionToken").setTerm(token.toString).setValue(1).build()
        ).toList

        // Generate userProfileFeatures
        val gender = userProfileFeatures.gender.toString
        val age = userProfileFeatures.age.toString
        val genderFeature = NameTermValueAvro.newBuilder().setName("gender").setTerm(gender).setValue(1).build()
        val ageFeature = NameTermValueAvro.newBuilder().setName("age").setTerm(age).setValue(1).build()
        val genderAgeFeature = NameTermValueAvro.newBuilder().setName("gender|age").setTerm(gender + "|" + age)
            .setValue(1).build()
        val userProfileFeaturesAvro = List(genderFeature, ageFeature, genderAgeFeature)

        val dataRecordBuffer = new ArrayBuffer[KDDCup2012DataAvro]()
        val clickGap = if (isTraining) click else 1
        val impressionGap = if (isTraining) impression - click else 1

        while (click > 0) {
          dataRecordBuffer += KDDCup2012DataAvro.newBuilder().setResponse(1).setWeight(clickGap).setOffset(0)
              .setAdID(adId).setAdvertiserID(advertiserId).setQueryID(queryId).setKeywordID(keywordId)
              .setTitleID(titleId).setDescriptionID(descriptionId).setUserID(userId)
              .setGlobalFeatures(globalFeaturesAvro).setQueryTokensFeatures(queryTokensFeaturesAvro)
              .setPurchasedKeywordTokensFeatures(purchasedKeywordTokensFeaturesAvro)
              .setTitleTokensFeatures(titleTokensFeaturesAvro)
              .setDescriptionTokensFeatures(descriptionTokensFeaturesAvro)
              .setUserProfileFeatures(userProfileFeaturesAvro)
              .build()
          click -= clickGap
          impression -= clickGap
        }

        while (impression > 0) {
          dataRecordBuffer += KDDCup2012DataAvro.newBuilder().setResponse(0).setWeight(impressionGap).setOffset(0)
              .setAdID(adId).setAdvertiserID(advertiserId).setQueryID(queryId).setKeywordID(keywordId)
              .setTitleID(titleId).setDescriptionID(descriptionId).setUserID(userId)
              .setGlobalFeatures(globalFeaturesAvro).setQueryTokensFeatures(queryTokensFeaturesAvro)
              .setPurchasedKeywordTokensFeatures(purchasedKeywordTokensFeaturesAvro)
              .setTitleTokensFeatures(titleTokensFeaturesAvro)
              .setDescriptionTokensFeatures(descriptionTokensFeaturesAvro)
              .setUserProfileFeatures(userProfileFeaturesAvro)
              .build()
          impression -= impressionGap
        }
        dataRecordBuffer.toArray[KDDCup2012DataAvro]
      }
      AvroIOUtils.saveAsAvro(records, outputPath.toString, KDDCup2012DataAvro.getClassSchema.toString)
    }

    val numExecutors = sparkContext.getExecutorStorageStatus.length

    Utils.deleteHDFSDir(trainingDataOutputPath.toString, hadoopConfiguration)
    joinDataWithFeaturesAndWriteAsAvroToHDFS(trainingDataInputPath, trainingDataOutputPath, isTraining = true,
      minPartitions = numExecutors * 2)
    println("Training data is ready")
    Utils.deleteHDFSDir(publicTestingDataOutputPath.toString, hadoopConfiguration)
    joinDataWithFeaturesAndWriteAsAvroToHDFS(publicTestingDataInputPath, publicTestingDataOutputPath,
      isTraining = false, minPartitions = numExecutors)
    println("Public testing data is ready")
    Utils.deleteHDFSDir(privateTestingDataOutputPath.toString, hadoopConfiguration)
    joinDataWithFeaturesAndWriteAsAvroToHDFS(privateTestingDataInputPath, privateTestingDataOutputPath,
      isTraining = false, minPartitions = numExecutors)
    println("Private testing data is ready")

    sparkContext.stop()
  }
}
