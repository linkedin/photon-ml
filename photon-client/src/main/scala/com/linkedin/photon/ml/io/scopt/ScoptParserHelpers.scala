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
package com.linkedin.photon.ml.io.scopt

import java.util.StringJoiner

import scala.util.Try
import scala.collection.mutable

import com.linkedin.photon.ml.Types.{CoordinateId, FeatureShardId}
import com.linkedin.photon.ml.data.{FixedEffectDataConfiguration, InputColumnsNames, RandomEffectDataConfiguration}
import com.linkedin.photon.ml.io.{CoordinateConfiguration, FeatureShardConfiguration, FixedEffectCoordinateConfiguration, RandomEffectCoordinateConfiguration}
import com.linkedin.photon.ml.optimization._
import com.linkedin.photon.ml.optimization.RegularizationType._
import com.linkedin.photon.ml.optimization.game.{FixedEffectOptimizationConfiguration, GLMOptimizationConfiguration, RandomEffectOptimizationConfiguration}
import com.linkedin.photon.ml.projector.IdentityProjection
import com.linkedin.photon.ml.util.{DoubleRange, Logging, PhotonLogger}

/**
 * Helper values/functions for parsing Scopt parameters.
 */
object ScoptParserHelpers extends Logging {

  //
  // Constants
  //

  val KV_DELIMITER = "="
  val LIST_DELIMITER = ","
  val SECONDARY_LIST_DELIMITER = '|'
  val RANGE_DELIMITER = '-'

  // Feature shard configuration parameters
  val FEATURE_SHARD_CONFIG_NAME = "name"
  val FEATURE_SHARD_CONFIG_FEATURE_BAGS = "feature.bags"
  val FEATURE_SHARD_CONFIG_INTERCEPT = "intercept"

  val FEATURE_SHARD_CONFIG_REQUIRED_ARGS = Map(
    (FEATURE_SHARD_CONFIG_NAME, "<name>"),
    (FEATURE_SHARD_CONFIG_FEATURE_BAGS,
      Seq("<featureBag1>", "<featureBag2>", "...").mkString(s"$SECONDARY_LIST_DELIMITER")))
  val FEATURE_SHARD_CONFIG_OPTIONAL_ARGS = Map((FEATURE_SHARD_CONFIG_INTERCEPT, "<bool>"))

  // Coordinate configuration parameters
  val COORDINATE_CONFIG_NAME = "name"

  val COORDINATE_DATA_CONFIG_RANDOM_EFFECT_TYPE = "random.effect.type"
  val COORDINATE_DATA_CONFIG_FEATURE_SHARD = "feature.shard"
  val COORDINATE_DATA_CONFIG_MIN_PARTITIONS = "min.partitions"
  val COORDINATE_DATA_CONFIG_ACTIVE_DATA_BOUND = "active.data.bound"
  val COORDINATE_DATA_CONFIG_PASSIVE_DATA_BOUND = "passive.data.bound"
  val COORDINATE_DATA_CONFIG_FEATURES_TO_SAMPLES_RATIO = "features.to.samples.ratio"

  val COORDINATE_OPT_CONFIG_OPTIMIZER = "optimizer"
  val COORDINATE_OPT_CONFIG_MAX_ITER = "max.iter"
  val COORDINATE_OPT_CONFIG_TOLERANCE = "tolerance"
  val COORDINATE_OPT_CONFIG_REGULARIZATION = "regularization"
  val COORDINATE_OPT_CONFIG_REG_ALPHA = "reg.alpha"
  val COORDINATE_OPT_CONFIG_REG_WEIGHTS = "reg.weights"
  val COORDINATE_OPT_CONFIG_DOWN_SAMPLING_RATE = "down.sampling.rate"

  val COORDINATE_CONFIG_REQUIRED_ARGS = Map(
    (COORDINATE_CONFIG_NAME, "<name>"),
    (COORDINATE_DATA_CONFIG_FEATURE_SHARD, "<shard>"),
    (COORDINATE_DATA_CONFIG_MIN_PARTITIONS, "<value>"),
    (COORDINATE_OPT_CONFIG_OPTIMIZER, s"[${OptimizerType.values.mkString(", ")}]"),
    (COORDINATE_OPT_CONFIG_MAX_ITER, "<value>"),
    (COORDINATE_OPT_CONFIG_TOLERANCE, "<value>"))
  val COORDINATE_CONFIG_OPTIONAL_ARGS = Map(
    (COORDINATE_DATA_CONFIG_RANDOM_EFFECT_TYPE, "<type>"),
    (COORDINATE_DATA_CONFIG_ACTIVE_DATA_BOUND, "<value>"),
    (COORDINATE_DATA_CONFIG_PASSIVE_DATA_BOUND, "<value>"),
    (COORDINATE_DATA_CONFIG_FEATURES_TO_SAMPLES_RATIO, "<value>"),
    (COORDINATE_OPT_CONFIG_REGULARIZATION, s"[${RegularizationType.values.mkString(", ")}]"),
    (COORDINATE_OPT_CONFIG_REG_ALPHA, "<value>"),
    (COORDINATE_OPT_CONFIG_REG_WEIGHTS, Seq("<value1>", "<value2>", "...").mkString(s"$SECONDARY_LIST_DELIMITER")),
    (COORDINATE_OPT_CONFIG_DOWN_SAMPLING_RATE, "<value>"))

  val COORDINATE_CONFIG_FIXED_ONLY_ARGS = Seq(COORDINATE_OPT_CONFIG_DOWN_SAMPLING_RATE)
  val COORDINATE_CONFIG_RANDOM_ONLY_ARGS = Seq(
    COORDINATE_DATA_CONFIG_ACTIVE_DATA_BOUND,
    COORDINATE_DATA_CONFIG_PASSIVE_DATA_BOUND,
    COORDINATE_DATA_CONFIG_FEATURES_TO_SAMPLES_RATIO)

  //
  // Parsing functions
  //

  /**
   * Create a [[Set]] from a [[Seq]].
   *
   * @tparam T The type of the values in the input/output collection
   * @param input A [[Seq]] of values.
   * @return A [[Set]] containing the unique input values
   */
  def parseSetFromSeq[T](input: Seq[T]): Set[T] = input.toSet

  /**
   * Create an [[InputColumnsNames]] instance from a [[Map]] of (default column name -> new column name).
   *
   * @param input A [[Map]] of (default column name -> new column name)
   * @return A new [[InputColumnsNames]] instance
   */
  def parseInputColumnNames(input: Map[String, String]): InputColumnsNames =
    input.foldLeft(InputColumnsNames()) { case (columns, (origColName, newColName)) =>
      columns.updated(InputColumnsNames.withName(origColName), newColName)
    }

  /**
   * Convert a log level [[String]] to a log level constant.
   *
   * @param input A log level [[String]]
   * @return The log level constant corresponding to the input
   */
  def parseLogLevel(input: String): Int = PhotonLogger.parseLogLevelString(input)

  /**
   * Create a single [[FeatureShardConfiguration]] from a [[Map]] of (feature shard arg -> feature shard value), and
   * stash it inside of a [[Map]].
   *
   * @param input A [[Map]] of (feature shard arg -> feature shard value)
   * @return A [[Map]] containing a single (feature shard name -> feature shard configuration) pair
   */
  def parseFeatureShardConfiguration(input: Map[String, String]): Map[FeatureShardId, FeatureShardConfiguration] = {

    val shardName = input(FEATURE_SHARD_CONFIG_NAME)
    val shardConfig = FeatureShardConfiguration(
      input(FEATURE_SHARD_CONFIG_FEATURE_BAGS).split(SECONDARY_LIST_DELIMITER).toSet,
      input.get(FEATURE_SHARD_CONFIG_INTERCEPT).map(_.toBoolean).getOrElse(true))

    Map((shardName, shardConfig))
  }

  /**
   * Create a [[DoubleRange]] from an input [[String]].
   *
   * @param input A [[DoubleRange]] in [[String]] format (two delimited [[Double]] values)
   * @return A [[DoubleRange]]
   * @throws IllegalArgumentException if the input cannot be parsed into delimited two [[Double]] values
   */
  def parseDoubleRange(input: String): DoubleRange =
    Try {
        val Array(start, end) = input.split(RANGE_DELIMITER)
        DoubleRange(start.toDouble, end.toDouble)
      }
      .getOrElse(throw new IllegalArgumentException(s"Couldn't parse the range '$input'."))

  /**
   * Create a single [[CoordinateConfiguration]] from a [[Map]] of (coordinate config arg -> coordinate config value),
   * and stash it inside of a [[Map]].
   *
   * @param input A [[Map]] of (coordinate config arg -> coordinate config value)
   * @return A [[Map]] containing a single (coordinate ID -> coordinate configuration) pair
   */
  def parseCoordinateConfiguration(input: Map[String, String]): Map[CoordinateId, CoordinateConfiguration] = {

    val coordinateName = input(COORDINATE_CONFIG_NAME)
    val featureShard = input(COORDINATE_DATA_CONFIG_FEATURE_SHARD)
    val minPartitions = input(COORDINATE_DATA_CONFIG_MIN_PARTITIONS).toInt

    val optimizer = OptimizerType.withName(input(COORDINATE_OPT_CONFIG_OPTIMIZER))
    val maxIter = input(COORDINATE_OPT_CONFIG_MAX_ITER).toInt
    val tolerance = input(COORDINATE_OPT_CONFIG_TOLERANCE).toDouble
    val optimizerConfig = OptimizerConfig(optimizer, maxIter, tolerance)

    val regularizationContext = input
      .get(COORDINATE_OPT_CONFIG_REGULARIZATION)
      .map { regularization =>
        RegularizationType.withName(regularization) match {
          case RegularizationType.L1 =>
            L1RegularizationContext

          case RegularizationType.L2 =>
            L2RegularizationContext

          case RegularizationType.ELASTIC_NET =>
            ElasticNetRegularizationContext(input(COORDINATE_OPT_CONFIG_REG_ALPHA).toDouble)

          case RegularizationType.NONE =>
            NoRegularizationContext
        }
      }
      .getOrElse(NoRegularizationContext)
    val regularizationWeights = input
      .get(COORDINATE_OPT_CONFIG_REG_WEIGHTS)
      .map(_.split(SECONDARY_LIST_DELIMITER).map(_.toDouble).toSet)
      .getOrElse(Set())

    val reTypeOpt = input.get(COORDINATE_DATA_CONFIG_RANDOM_EFFECT_TYPE)
    val config = reTypeOpt match {
      // Random effect coordinate
      case Some(reType) =>
        val dataConfig = RandomEffectDataConfiguration(
          reType,
          featureShard,
          minPartitions,
          input.get(COORDINATE_DATA_CONFIG_ACTIVE_DATA_BOUND).map(_.toInt),
          input.get(COORDINATE_DATA_CONFIG_PASSIVE_DATA_BOUND).map(_.toInt),
          input.get(COORDINATE_DATA_CONFIG_FEATURES_TO_SAMPLES_RATIO).map(_.toDouble),
          IdentityProjection)
        val optConfig = RandomEffectOptimizationConfiguration(optimizerConfig, regularizationContext)

        //
        COORDINATE_CONFIG_FIXED_ONLY_ARGS.foreach { config =>
          input.get(config).foreach { _ =>
            logger.warn(s"Found and ignored $config for random effect coordinate '$coordinateName'")
          }
        }

        RandomEffectCoordinateConfiguration(dataConfig, optConfig, regularizationWeights)

      case None =>
        val dataConfig = FixedEffectDataConfiguration(featureShard, minPartitions)
        val optConfig = FixedEffectOptimizationConfiguration(optimizerConfig, regularizationContext)

        //
        COORDINATE_CONFIG_RANDOM_ONLY_ARGS.foreach { config =>
          input.get(config).foreach { _ =>
            logger.warn(s"Found and ignored $config for fixed effect coordinate '$coordinateName'")
          }
        }

        FixedEffectCoordinateConfiguration(
          dataConfig,
          input
            .get(COORDINATE_OPT_CONFIG_DOWN_SAMPLING_RATE)
            .map(rate => optConfig.copy(downSamplingRate = rate.toDouble))
            .getOrElse(optConfig),
          regularizationWeights)
    }

    Map((coordinateName, config))
  }

  //
  // Update functions
  //

  /**
   * Add a [[FeatureShardConfiguration]] to an existing [[Map]] of [[FeatureShardConfiguration]].
   *
   * @param newConfig A single (feature shard ID -> feature shard configuration) pair, wrapped in a [[Map]]
   * @param existingConfigs A [[Map]] of (feature shard ID -> feature shard configuration) pairs
   * @return The [[Map]] of existing [[FeatureShardConfiguration]] with the new one added in
   */
  def updateFeatureShardConfigurations(
      newConfig: Map[FeatureShardId, FeatureShardConfiguration],
      existingConfigs: Map[FeatureShardId, FeatureShardConfiguration]): Map[FeatureShardId, FeatureShardConfiguration] = {

    val configPair = newConfig.iterator.next()
    val featureShardName = configPair._1

    existingConfigs.get(featureShardName) match {
      case Some(_) =>
        throw new IllegalArgumentException(s"Feature shard config '$featureShardName' defined more than once")

      case None =>
        existingConfigs + configPair
    }
  }

  /**
   * Add a [[CoordinateConfiguration]] to an existing [[Map]] of [[CoordinateConfiguration]].
   *
   * @param newConfig A single (coordinate ID -> coordinate configuration) pair, wrapped in a [[Map]]
   * @param existingConfigs A [[Map]] of (coordinate ID -> coordinate configuration) pairs
   * @return The [[Map]] of existing [[CoordinateConfiguration]] with the new one added in
   */
  def updateCoordinateConfigurations(
    newConfig: Map[CoordinateId, CoordinateConfiguration],
    existingConfigs: Map[CoordinateId, CoordinateConfiguration]): Map[CoordinateId, CoordinateConfiguration] = {

    val configPair = newConfig.iterator.next()
    val coordinateName = configPair._1

    existingConfigs.get(coordinateName) match {
      case Some(_) =>
        throw new IllegalArgumentException(s"Feature shard config '$coordinateName' defined more than once")

      case None =>
        existingConfigs + configPair
    }
  }

  //
  // Printing functions
  //

  /**
   * Convert a [[Set]] of values to a [[String]] parseable by Scopt.
   *
   * @tparam T The type of the values in the [[Set]]
   * @param output A [[Set]] of values
   * @return A [[String]] of the [[Set]] values joined by the Scopt list delimiter
   */
  def setToString[T](output: Set[T]): String = output.map(_.toString).mkString(LIST_DELIMITER)

  /**
   * Convert a [[Seq]] of values to a [[String]] parseable by Scopt.
   *
   * @tparam T The type of the values in the [[Seq]]
   * @param output A [[Seq]] of values
   * @return A [[String]] of the [[Seq]] values joined by the Scopt list delimiter
   */
  def seqToString[T](output: Seq[T]): String = output.map(_.toString).mkString(LIST_DELIMITER)

  /**
   * Convert an [[InputColumnsNames]] instance to a [[String]] parseable by Scopt.
   *
   * @param output An [[InputColumnsNames]] instance
   * @return A [[String]] of Scopt map pairs (original column name -> new column name) joined by the Scopt list
   *         delimiter
   */
  def inputColumnNamesToString(output: InputColumnsNames): String =
    InputColumnsNames.all.map(icn => s"$icn$KV_DELIMITER${output(icn)}").mkString(LIST_DELIMITER)

  /**
   * Convert a [[Map]] of (feature shard ID -> feature shard config) pairs to a list of [[String]] parseable by Scopt.
   *
   * @param output a [[Map]] of (feature shard ID -> feature shard config) pairs
   * @return A [[Seq]] of [[String]], each one containing Scopt map pairs (feature shard arg -> feature shard value)
   *         joined by the Scopt list delimiter
   */
  def featureShardConfigsToStrings(output: Map[FeatureShardId, FeatureShardConfiguration]): Seq[String] = {
    output.toSeq.map { case (featureShardName, featureShardConfig) =>
      val strJoiner = new StringJoiner(LIST_DELIMITER)
      val argsMap = mutable.LinkedHashMap[String, String]()

      //
      // Append required args
      //

      argsMap += (FEATURE_SHARD_CONFIG_NAME -> featureShardName)
      argsMap +=
        (FEATURE_SHARD_CONFIG_FEATURE_BAGS -> featureShardConfig.featureBags.mkString(s"$SECONDARY_LIST_DELIMITER"))

      //
      // Append optional args
      //

      if (!featureShardConfig.hasIntercept) {
        argsMap += (FEATURE_SHARD_CONFIG_INTERCEPT -> featureShardConfig.hasIntercept.toString)
      }

      //
      // Build feature shard config args string
      //

      argsMap.foreach { case (arg, value) =>
        strJoiner.add(arg + KV_DELIMITER + value)
      }
      strJoiner.toString
    }
  }

  /**
   * Convert a [[DoubleRange]] instance to a [[String]] parseable by Scopt.
   *
   * @param output A [[DoubleRange]] instance
   * @return A [[String]] of the delimited [[DoubleRange]] bounds
   */
  def doubleRangeToString(output: DoubleRange): String = s"${output.start}$RANGE_DELIMITER${output.end}"

  /**
   * Convert a [[Map]] of (coordinate ID -> coordinate config) pairs to a list of [[String]] parseable by Scopt.
   *
   * @param output a [[Map]] of (coordinate ID -> coordinate config) pairs
   * @return A [[Seq]] of [[String]], each one containing Scopt map pairs (coordinate arg -> coordinate value)
   *         joined by the Scopt list delimiter
   */
  def coordinateConfigsToStrings(output: Map[CoordinateId, CoordinateConfiguration]): Seq[String] = {
    output.toSeq.map { case (coordinateName, coordinateConfig) =>
      val strJoiner = new StringJoiner(LIST_DELIMITER)
      val argsMap = mutable.LinkedHashMap[String, String]()
      val dataConfig = coordinateConfig.dataConfiguration
      val optConfig = coordinateConfig.optimizationConfiguration.asInstanceOf[GLMOptimizationConfiguration]

      //
      // Append required args
      //

      argsMap += (COORDINATE_CONFIG_NAME -> coordinateName)

      argsMap += (COORDINATE_DATA_CONFIG_FEATURE_SHARD -> dataConfig.featureShardId)
      argsMap += (COORDINATE_DATA_CONFIG_MIN_PARTITIONS -> dataConfig.minNumPartitions.toString)

      argsMap += (COORDINATE_OPT_CONFIG_OPTIMIZER -> optConfig.optimizerConfig.optimizerType.toString)
      argsMap += (COORDINATE_OPT_CONFIG_TOLERANCE -> optConfig.optimizerConfig.tolerance.toString)
      argsMap += (COORDINATE_OPT_CONFIG_MAX_ITER -> optConfig.optimizerConfig.maximumIterations.toString)

      //
      // Append optional args
      //

      dataConfig match {
        case reDataConfig: RandomEffectDataConfiguration =>
          argsMap += (COORDINATE_DATA_CONFIG_RANDOM_EFFECT_TYPE -> reDataConfig.randomEffectType)

          reDataConfig.numActiveDataPointsUpperBound.foreach { bound =>
            argsMap += (COORDINATE_DATA_CONFIG_ACTIVE_DATA_BOUND -> bound.toString)
          }
          reDataConfig.numPassiveDataPointsLowerBound.foreach { bound =>
            argsMap += (COORDINATE_DATA_CONFIG_PASSIVE_DATA_BOUND -> bound.toString)
          }
          reDataConfig.numFeaturesToSamplesRatioUpperBound.foreach { ratio =>
            argsMap += (COORDINATE_DATA_CONFIG_FEATURES_TO_SAMPLES_RATIO -> ratio.toString)
          }

        case _ =>
      }

      optConfig match {
        case feOptConfig: FixedEffectOptimizationConfiguration =>
          argsMap += (COORDINATE_OPT_CONFIG_DOWN_SAMPLING_RATE -> feOptConfig.downSamplingRate.toString)

        case _ =>
      }

      optConfig.regularizationContext.regularizationType match {
        case NONE =>

        case regType: RegularizationType =>
          argsMap += (COORDINATE_OPT_CONFIG_REGULARIZATION -> regType.toString)
          optConfig.regularizationContext.elasticNetParam.foreach { alpha =>
            argsMap += (COORDINATE_OPT_CONFIG_REG_ALPHA -> alpha.toString)
          }
          argsMap +=
            (COORDINATE_OPT_CONFIG_REG_WEIGHTS ->
              coordinateConfig.regularizationWeights.mkString(s"$SECONDARY_LIST_DELIMITER"))
      }

      //
      // Build feature shard config args string
      //

      argsMap.foreach { case (arg, value) =>
        strJoiner.add(arg + KV_DELIMITER + value)
      }
      strJoiner.toString
    }
  }
}
