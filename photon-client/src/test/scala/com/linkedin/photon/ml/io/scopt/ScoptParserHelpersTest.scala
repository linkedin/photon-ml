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

import scala.collection.SortedSet
import scala.collection.immutable.ListMap

import org.mockito.Mockito._
import org.testng.Assert._
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.Types.{CoordinateId, FeatureShardId}
import com.linkedin.photon.ml.data.{FixedEffectDataConfiguration, InputColumnsNames, RandomEffectDataConfiguration}
import com.linkedin.photon.ml.io.{CoordinateConfiguration, FeatureShardConfiguration, FixedEffectCoordinateConfiguration, RandomEffectCoordinateConfiguration}
import com.linkedin.photon.ml.optimization.game.{FixedEffectOptimizationConfiguration, RandomEffectOptimizationConfiguration}
import com.linkedin.photon.ml.optimization._
import com.linkedin.photon.ml.projector.IdentityProjection
import com.linkedin.photon.ml.util.DoubleRange

/**
 * Unit tests for [[ScoptParserHelpers]].
 */
class ScoptParserHelpersTest {

  /**
   * Test that an [[InputColumnsNames]] instance can be correctly parsed.
   */
  @Test
  def testParseInputColumnNames(): Unit = {

    val newNames = InputColumnsNames.all.map(_.toString).zipWithIndex.toMap.mapValues(_.toString)
    val newInputCols = ScoptParserHelpers.parseInputColumnNames(newNames)

    InputColumnsNames.all.foreach { column =>
      assertEquals(newInputCols(column), newNames(column.toString))
    }
  }

  /**
   * Test that a [[FeatureShardConfiguration]] instance can be correctly parsed.
   */
  @Test
  def testParseFeatureShardConfiguration(): Unit = {

    val featureShardId1 = "shard1"
    val featureBagsStr1 = "unique"
    val featureBags1 = Set("unique")
    val featureShardIntercept1 = true
    val inputMap1 = Map[String, String](
      ScoptParserHelpers.FEATURE_SHARD_CONFIG_NAME -> featureShardId1,
      ScoptParserHelpers.FEATURE_SHARD_CONFIG_FEATURE_BAGS -> featureBagsStr1)

    val featureShardId2 = "shard2"
    val featureBagsStr2 = Seq("unique", "repeat", "repeat").mkString(s"${ScoptParserHelpers.SECONDARY_LIST_DELIMITER}")
    val featureBags2 = Set("unique", "repeat")
    val featureShardIntercept2 = false
    val inputMap2 = Map[String, String](
      ScoptParserHelpers.FEATURE_SHARD_CONFIG_NAME -> featureShardId2,
      ScoptParserHelpers.FEATURE_SHARD_CONFIG_FEATURE_BAGS -> featureBagsStr2,
      ScoptParserHelpers.FEATURE_SHARD_CONFIG_INTERCEPT -> featureShardIntercept2.toString)

    val featureShardConfigs = ScoptParserHelpers.parseFeatureShardConfiguration(inputMap1) ++
      ScoptParserHelpers.parseFeatureShardConfiguration(inputMap2)

    assertEquals(featureShardConfigs(featureShardId1).featureBags, featureBags1)
    assertEquals(featureShardConfigs(featureShardId1).hasIntercept, featureShardIntercept1)

    assertEquals(featureShardConfigs(featureShardId2).featureBags, featureBags2)
    assertEquals(featureShardConfigs(featureShardId2).hasIntercept, featureShardIntercept2)
  }

  /**
   * Test that a [[CoordinateConfiguration]] instance can be correctly parsed.
   */
  @Test
  def testParseCoordinateConfiguration(): Unit = {

    val coordinateId = "coordinate"
    val featureShard = "shard"
    val minPartitions = 1
    val optimizer = OptimizerType.LBFGS
    val maxIter = 2
    val tolerance = 3e-3

    val alpha = 0.4
    val reType = "type"
    val regularization1 = NoRegularizationContext
    val regularizationType = RegularizationType.ELASTIC_NET
    val regularization2 = RegularizationContext(regularizationType, Some(alpha))
    val regWeights1 = Set()
    val regWeights2Str = Seq("1", "10", "100", "100", "10").mkString(s"${ScoptParserHelpers.SECONDARY_LIST_DELIMITER}")
    val regWeights2 = Set(1, 10, 100)
    val activeDataUpperBound1 = None
    val activeDataUpperBound2 = Some(5)
    val passiveDataLowerBound1 = None
    val passiveDataLowerBound2 = Some(6)
    val featuresSamplesRatio1 = None
    val featuresSamplesRatio2 = Some(7)
    val downSamplingRate1 = 1.0
    val downSamplingRate2 = 0.7

    val inputMap1 = Map[String, String](
      ScoptParserHelpers.COORDINATE_CONFIG_NAME -> coordinateId,
      ScoptParserHelpers.COORDINATE_DATA_CONFIG_FEATURE_SHARD -> featureShard,
      ScoptParserHelpers.COORDINATE_DATA_CONFIG_MIN_PARTITIONS -> minPartitions.toString,
      ScoptParserHelpers.COORDINATE_OPT_CONFIG_OPTIMIZER -> optimizer.toString,
      ScoptParserHelpers.COORDINATE_OPT_CONFIG_MAX_ITER -> maxIter.toString,
      ScoptParserHelpers.COORDINATE_OPT_CONFIG_TOLERANCE -> tolerance.toString)
    val inputMap2 = inputMap1 ++ Map[String, String](
      ScoptParserHelpers.COORDINATE_OPT_CONFIG_DOWN_SAMPLING_RATE -> downSamplingRate2.toString,
      ScoptParserHelpers.COORDINATE_OPT_CONFIG_REGULARIZATION -> regularizationType.toString,
      ScoptParserHelpers.COORDINATE_OPT_CONFIG_REG_WEIGHTS -> regWeights2Str,
      ScoptParserHelpers.COORDINATE_OPT_CONFIG_REG_ALPHA -> alpha.toString)
    val inputMap3 = inputMap1 ++ Map[String, String](
      ScoptParserHelpers.COORDINATE_DATA_CONFIG_RANDOM_EFFECT_TYPE -> reType)
    val inputMap4 = inputMap1 ++ Map[String, String](
      ScoptParserHelpers.COORDINATE_DATA_CONFIG_RANDOM_EFFECT_TYPE -> reType,
      ScoptParserHelpers.COORDINATE_DATA_CONFIG_ACTIVE_DATA_BOUND -> activeDataUpperBound2.get.toString,
      ScoptParserHelpers.COORDINATE_DATA_CONFIG_PASSIVE_DATA_BOUND -> passiveDataLowerBound2.get.toString,
      ScoptParserHelpers.COORDINATE_DATA_CONFIG_FEATURES_TO_SAMPLES_RATIO -> featuresSamplesRatio2.get.toString,
      ScoptParserHelpers.COORDINATE_OPT_CONFIG_REGULARIZATION -> regularizationType.toString,
      ScoptParserHelpers.COORDINATE_OPT_CONFIG_REG_WEIGHTS -> regWeights2Str,
      ScoptParserHelpers.COORDINATE_OPT_CONFIG_REG_ALPHA -> alpha.toString)

    val coordinateConfig1 = ScoptParserHelpers.parseCoordinateConfiguration(inputMap1)(coordinateId)
    coordinateConfig1 match {
      case feConfig: FixedEffectCoordinateConfiguration =>
        assertEquals(feConfig.dataConfiguration.featureShardId, featureShard)
        assertEquals(feConfig.dataConfiguration.minNumPartitions, minPartitions)
        assertEquals(feConfig.optimizationConfiguration.optimizerConfig.optimizerType, optimizer)
        assertEquals(feConfig.optimizationConfiguration.optimizerConfig.maximumIterations, maxIter)
        assertEquals(feConfig.optimizationConfiguration.optimizerConfig.tolerance, tolerance)
        assertEquals(feConfig.optimizationConfiguration.regularizationContext, regularization1)
        assertEquals(feConfig.optimizationConfiguration.downSamplingRate, downSamplingRate1)
        assertEquals(feConfig.regularizationWeights, regWeights1)

      case _ =>
        throw new IllegalArgumentException("Expected fixed effect coordinate configuration.")
    }

    val coordinateConfig2 = ScoptParserHelpers.parseCoordinateConfiguration(inputMap2)(coordinateId)
    coordinateConfig2 match {
      case feConfig: FixedEffectCoordinateConfiguration =>
        assertEquals(feConfig.optimizationConfiguration.regularizationContext, regularization2)
        assertEquals(feConfig.optimizationConfiguration.downSamplingRate, downSamplingRate2)
        assertEquals(feConfig.regularizationWeights, regWeights2)

      case _ =>
        throw new IllegalArgumentException("Expected fixed effect coordinate configuration.")
    }

    val coordinateConfig3 = ScoptParserHelpers.parseCoordinateConfiguration(inputMap3)(coordinateId)
    coordinateConfig3 match {
      case reConfig: RandomEffectCoordinateConfiguration =>
        assertEquals(reConfig.dataConfiguration.randomEffectType, reType)
        assertEquals(reConfig.dataConfiguration.featureShardId, featureShard)
        assertEquals(reConfig.dataConfiguration.minNumPartitions, minPartitions)
        assertEquals(reConfig.dataConfiguration.numActiveDataPointsUpperBound, activeDataUpperBound1)
        assertEquals(reConfig.dataConfiguration.numPassiveDataPointsLowerBound, passiveDataLowerBound1)
        assertEquals(reConfig.dataConfiguration.numFeaturesToSamplesRatioUpperBound, featuresSamplesRatio1)
        assertEquals(reConfig.optimizationConfiguration.optimizerConfig.optimizerType, optimizer)
        assertEquals(reConfig.optimizationConfiguration.optimizerConfig.maximumIterations, maxIter)
        assertEquals(reConfig.optimizationConfiguration.optimizerConfig.tolerance, tolerance)
        assertEquals(reConfig.optimizationConfiguration.regularizationContext, regularization1)
        assertEquals(reConfig.regularizationWeights, regWeights1)

      case _ =>
        throw new IllegalArgumentException("Expected random effect coordinate configuration.")
    }

    val coordinateConfig4 = ScoptParserHelpers.parseCoordinateConfiguration(inputMap4)(coordinateId)
    coordinateConfig4 match {
      case reConfig: RandomEffectCoordinateConfiguration =>
        assertEquals(reConfig.dataConfiguration.numActiveDataPointsUpperBound, activeDataUpperBound2)
        assertEquals(reConfig.dataConfiguration.numPassiveDataPointsLowerBound, passiveDataLowerBound2)
        assertEquals(reConfig.dataConfiguration.numFeaturesToSamplesRatioUpperBound, featuresSamplesRatio2)
        assertEquals(reConfig.optimizationConfiguration.regularizationContext, regularization2)
        assertEquals(reConfig.regularizationWeights, regWeights2)

      case _ =>
        throw new IllegalArgumentException("Expected random effect coordinate configuration.")
    }
  }

  /**
   * Test that an existing feature shard config map can be updated with another [[FeatureShardConfiguration]].
   */
  @Test
  def testUpdateFeatureShardConfiguration(): Unit = {

    val featureShard1 = "featureShard1"
    val featureShard2 = "featureShard2"
    val config1 = mock(classOf[FeatureShardConfiguration])
    val config2 = mock(classOf[FeatureShardConfiguration])
    val map1 = Map[FeatureShardId, FeatureShardConfiguration](featureShard1 -> config1)
    val map2 = Map[FeatureShardId, FeatureShardConfiguration](featureShard2 -> config2)
    val combinedMap = ScoptParserHelpers.updateFeatureShardConfigurations(map1, map2)

    assertTrue(combinedMap.contains(featureShard1))
    assertTrue(combinedMap.contains(featureShard2))
    assertEquals(combinedMap(featureShard1), config1)
    assertEquals(combinedMap(featureShard2), config2)
  }

  @DataProvider
  def invalidFeatureShardConfigurations(): Array[Array[Any]] = {

    val featureShard = "featureShard"
    val config1 = mock(classOf[FeatureShardConfiguration])
    val config2 = mock(classOf[FeatureShardConfiguration])
    val map1 = Map[FeatureShardId, FeatureShardConfiguration]()
    val map2 = Map[FeatureShardId, FeatureShardConfiguration](featureShard -> config1)
    val map3 = Map[FeatureShardId, FeatureShardConfiguration](featureShard -> config2)

    Array(
      Array(map1, map2),
      Array(map2, map3))
  }

  /**
   * Test that an empty or existing [[FeatureShardConfiguration]] cannot be used to update the feature shard config map.
   *
   * @param newMap An invalid feature shard config map
   * @param existingMap The existing feature shard config map
   */
  @Test(
    dataProvider = "invalidFeatureShardConfigurations",
    expectedExceptions = Array(classOf[IllegalArgumentException], classOf[NoSuchElementException]))
  def testInvalidUpdateFeatureShardConfiguration(
      newMap: Map[FeatureShardId, FeatureShardConfiguration],
      existingMap: Map[FeatureShardId, FeatureShardConfiguration]): Unit =
    ScoptParserHelpers.updateFeatureShardConfigurations(newMap, existingMap)

  /**
   * Test that an existing coordinate config map can be updated with another [[CoordinateConfiguration]].
   */
  @Test
  def testUpdateCoordinateConfiguration(): Unit = {

    val coordinate1 = "coordinate1"
    val coordinate2 = "coordinate2"
    val config1 = mock(classOf[CoordinateConfiguration])
    val config2 = mock(classOf[CoordinateConfiguration])
    val map1 = Map[CoordinateId, CoordinateConfiguration](coordinate1 -> config1)
    val map2 = Map[CoordinateId, CoordinateConfiguration](coordinate2 -> config2)
    val combinedMap = ScoptParserHelpers.updateCoordinateConfigurations(map1, map2)

    assertTrue(combinedMap.contains(coordinate1))
    assertTrue(combinedMap.contains(coordinate2))
    assertEquals(combinedMap(coordinate1), config1)
    assertEquals(combinedMap(coordinate2), config2)
  }

  @DataProvider
  def invalidCoordinateConfigurations(): Array[Array[Any]] = {

    val coordinate = "coordinate"
    val config1 = mock(classOf[CoordinateConfiguration])
    val config2 = mock(classOf[CoordinateConfiguration])
    val map1 = Map[CoordinateId, CoordinateConfiguration]()
    val map2 = Map[CoordinateId, CoordinateConfiguration](coordinate -> config1)
    val map3 = Map[CoordinateId, CoordinateConfiguration](coordinate -> config2)

    Array(
      Array(map1, map2),
      Array(map2, map3))
  }

  /**
   * Test that an empty or existing [[CoordinateConfiguration]] cannot be used to update the coordinate config map.
   *
   * @param newMap An invalid coordinate config map
   * @param existingMap The existing coordinate config map
   */
  @Test(
    dataProvider = "invalidCoordinateConfigurations",
    expectedExceptions = Array(classOf[IllegalArgumentException], classOf[NoSuchElementException]))
  def testInvalidUpdateCoordinateConfiguration(
      newMap: Map[CoordinateId, CoordinateConfiguration],
      existingMap: Map[CoordinateId, CoordinateConfiguration]): Unit =
    ScoptParserHelpers.updateCoordinateConfigurations(newMap, existingMap)

  /**
   * Test that a [[Set]] of values can be correctly printed into a Scopt-parseable [[String]].
   */
  @Test
  def testSetToString(): Unit = {

    val setOfInt = SortedSet[Int](1, 2, 3)
    val setOfString = setOfInt.map(_.toString)
    val expectedString = setOfString.mkString(ScoptParserHelpers.LIST_DELIMITER)

    assertEquals(ScoptParserHelpers.setToString(setOfString.asInstanceOf[Set[String]]), expectedString)
    assertEquals(ScoptParserHelpers.setToString(setOfInt.asInstanceOf[Set[Int]]), expectedString)
  }

  /**
   * Test that a [[Seq]] of values can be correctly printed into a Scopt-parseable [[String]].
   */
  @Test
  def testSeqToString(): Unit = {

    val seqOfInt = Seq[Int](1, 2, 3)
    val seqOfString = seqOfInt.map(_.toString)
    val expectedString = seqOfString.mkString(ScoptParserHelpers.LIST_DELIMITER)

    assertEquals(ScoptParserHelpers.seqToString(seqOfString), expectedString)
    assertEquals(ScoptParserHelpers.seqToString(seqOfInt), expectedString)
  }

  /**
   * Test that an [[InputColumnsNames]] instance can be correctly printed into a Scopt-parseable [[String]].
   */
  @Test
  def testInputColumnNamesToString(): Unit = {

    val columnNamesMap = InputColumnsNames.all.zipWithIndex.toMap.mapValues(_.toString)
    val inputColumnsNames = InputColumnsNames()
    val expected = columnNamesMap
      .map { case (column, name) =>
        s"$column${ScoptParserHelpers.KV_DELIMITER}$name"
      }
      .mkString(ScoptParserHelpers.LIST_DELIMITER)

    columnNamesMap.foreach { case (column, name) =>
      inputColumnsNames.updated(column, name)
    }

    assertEquals(ScoptParserHelpers.inputColumnNamesToString(inputColumnsNames), expected)
  }

  /**
   * Test that a multiple [[FeatureShardConfiguration]] instances can be correctly printed into a Scopt-parseable
   * [[String]].
   */
  @Test
  def testFeatureShardConfigsToStrings(): Unit = {

    val featureShardId1 = "featureShard1"
    val featureBags1 = SortedSet[FeatureShardId]("first").asInstanceOf[Set[String]]
    val featureShardIntercept1 = false
    val featureShardConfig1 = FeatureShardConfiguration(featureBags1, featureShardIntercept1)

    val featureShardId2 = "featureShard2"
    val featureBags2 = SortedSet[FeatureShardId]("first", "second", "third").asInstanceOf[Set[String]]
    val featureShardIntercept2 = true
    val featureShardConfig2 = FeatureShardConfiguration(featureBags2, featureShardIntercept2)

    val featureShardConfigurations = ListMap[FeatureShardId, FeatureShardConfiguration](
      featureShardId1 -> featureShardConfig1,
      featureShardId2 -> featureShardConfig2)

    val expected1 = Seq[(String, String)](
        ScoptParserHelpers.FEATURE_SHARD_CONFIG_NAME -> featureShardId1,
        ScoptParserHelpers.FEATURE_SHARD_CONFIG_FEATURE_BAGS -> featureBags1.head,
        ScoptParserHelpers.FEATURE_SHARD_CONFIG_INTERCEPT -> featureShardIntercept1.toString)
      .map { case (arg, value) =>
        s"$arg${ScoptParserHelpers.KV_DELIMITER}$value"
      }
      .mkString(ScoptParserHelpers.LIST_DELIMITER)
    val expected2 = Seq[(String, String)](
        ScoptParserHelpers.FEATURE_SHARD_CONFIG_NAME -> featureShardId2,
        ScoptParserHelpers.FEATURE_SHARD_CONFIG_FEATURE_BAGS ->
          featureBags2.mkString(s"${ScoptParserHelpers.SECONDARY_LIST_DELIMITER}"))
      .map { case (arg, value) =>
        s"$arg${ScoptParserHelpers.KV_DELIMITER}$value"
      }
      .mkString(ScoptParserHelpers.LIST_DELIMITER)

    val Seq(actual1, actual2) = ScoptParserHelpers.featureShardConfigsToStrings(featureShardConfigurations)

    assertEquals(actual1, expected1)
    assertEquals(actual2, expected2)
  }

  /**
   * Test that a [[DoubleRange]] instance can be correctly printed into a Scopt-parseable [[String]].
   */
  @Test
  def testDoubleRangeToString(): Unit = {

    val doubleRange = DoubleRange(1.1, 2.2)
    val expected = s"${doubleRange.start}${ScoptParserHelpers.RANGE_DELIMITER}${doubleRange.end}"

    assertEquals(ScoptParserHelpers.doubleRangeToString(doubleRange), expected)
  }

  /**
   * Test that a multiple [[CoordinateConfiguration]] instances can be correctly printed into a Scopt-parseable
   * [[String]].
   */
  @Test
  def testCoordinateConfigsToStrings(): Unit = {

    val featureShardId = "featureShard"
    val minPartitions = 1
    val optimizer = OptimizerType.LBFGS
    val tolerance = 2e-2
    val maxIterations = 3
    val optimizerConfig = OptimizerConfig(optimizer, maxIterations, tolerance)
    val reType = "type"
    val activeDataBound = 4
    val passiveDataBound = 5
    val featuresSamplesRatio = 6.0
    val regularizationType = RegularizationType.ELASTIC_NET
    val regularizationAlpha = 0.7
    val regularizationWeights = SortedSet[Double](8.8, 9.9).asInstanceOf[Set[Double]]

    val feDataConfig = FixedEffectDataConfiguration(featureShardId, minPartitions)

    val coordinateId1 = "coordinate1"
    val downSamplingRate1 = 1.0
    val optConfig1 = FixedEffectOptimizationConfiguration(optimizerConfig)
    val coordinateConfig1 = FixedEffectCoordinateConfiguration(feDataConfig, optConfig1)

    val coordinateId2 = "coordinate2"
    val downSamplingRate2 = 0.5
    val optConfig2 = FixedEffectOptimizationConfiguration(optimizerConfig, downSamplingRate = downSamplingRate2)
    val coordinateConfig2 = FixedEffectCoordinateConfiguration(feDataConfig, optConfig2)

    val coordinateId3 = "coordinate3"
    val dataConfig3 = RandomEffectDataConfiguration(
      reType,
      featureShardId,
      minPartitions,
      None,
      None,
      None,
      IdentityProjection)
    val optConfig3 = RandomEffectOptimizationConfiguration(optimizerConfig)
    val coordinateConfig3 = RandomEffectCoordinateConfiguration(dataConfig3, optConfig3)

    val coordinateId4 = "coordinate4"
    val dataConfig4 = RandomEffectDataConfiguration(
      reType,
      featureShardId,
      minPartitions,
      Some(activeDataBound),
      Some(passiveDataBound),
      Some(featuresSamplesRatio),
      IdentityProjection)
    val optConfig4 = RandomEffectOptimizationConfiguration(
      optimizerConfig,
      ElasticNetRegularizationContext(regularizationAlpha))
    val coordinateConfig4 = RandomEffectCoordinateConfiguration(dataConfig4, optConfig4, regularizationWeights)

    val coordinateConfigurations = ListMap[CoordinateId, CoordinateConfiguration](
      coordinateId1 -> coordinateConfig1,
      coordinateId2 -> coordinateConfig2,
      coordinateId3 -> coordinateConfig3,
      coordinateId4 -> coordinateConfig4)

    val baseArgs = Seq[(String, String)](
      ScoptParserHelpers.COORDINATE_DATA_CONFIG_FEATURE_SHARD -> featureShardId,
      ScoptParserHelpers.COORDINATE_DATA_CONFIG_MIN_PARTITIONS -> minPartitions.toString,
      ScoptParserHelpers.COORDINATE_OPT_CONFIG_OPTIMIZER -> optimizer.toString,
      ScoptParserHelpers.COORDINATE_OPT_CONFIG_TOLERANCE -> tolerance.toString,
      ScoptParserHelpers.COORDINATE_OPT_CONFIG_MAX_ITER -> maxIterations.toString)

    val expected1 = (((ScoptParserHelpers.COORDINATE_CONFIG_NAME -> coordinateId1) +: baseArgs) ++
      Seq[(String, String)](ScoptParserHelpers.COORDINATE_OPT_CONFIG_DOWN_SAMPLING_RATE -> downSamplingRate1.toString))
      .map { case (arg, value) =>
        s"$arg${ScoptParserHelpers.KV_DELIMITER}$value"
      }
      .mkString(ScoptParserHelpers.LIST_DELIMITER)
    val expected2 = (((ScoptParserHelpers.COORDINATE_CONFIG_NAME -> coordinateId2) +: baseArgs) ++
      Seq[(String, String)](ScoptParserHelpers.COORDINATE_OPT_CONFIG_DOWN_SAMPLING_RATE -> downSamplingRate2.toString))
      .map { case (arg, value) =>
        s"$arg${ScoptParserHelpers.KV_DELIMITER}$value"
      }
      .mkString(ScoptParserHelpers.LIST_DELIMITER)
    val expected3 = (((ScoptParserHelpers.COORDINATE_CONFIG_NAME -> coordinateId3) +: baseArgs) ++
      Seq[(String, String)](ScoptParserHelpers.COORDINATE_DATA_CONFIG_RANDOM_EFFECT_TYPE -> reType))
      .map { case (arg, value) =>
        s"$arg${ScoptParserHelpers.KV_DELIMITER}$value"
      }
      .mkString(ScoptParserHelpers.LIST_DELIMITER)
    val expected4 = (((ScoptParserHelpers.COORDINATE_CONFIG_NAME -> coordinateId4) +: baseArgs) ++
      Seq[(String, String)](
        ScoptParserHelpers.COORDINATE_DATA_CONFIG_RANDOM_EFFECT_TYPE -> reType,
        ScoptParserHelpers.COORDINATE_DATA_CONFIG_ACTIVE_DATA_BOUND -> activeDataBound.toString,
        ScoptParserHelpers.COORDINATE_DATA_CONFIG_PASSIVE_DATA_BOUND -> passiveDataBound.toString,
        ScoptParserHelpers.COORDINATE_DATA_CONFIG_FEATURES_TO_SAMPLES_RATIO -> featuresSamplesRatio.toString,
        ScoptParserHelpers.COORDINATE_OPT_CONFIG_REGULARIZATION -> regularizationType.toString,
        ScoptParserHelpers.COORDINATE_OPT_CONFIG_REG_ALPHA -> regularizationAlpha.toString,
        ScoptParserHelpers.COORDINATE_OPT_CONFIG_REG_WEIGHTS ->
          regularizationWeights.mkString(s"${ScoptParserHelpers.SECONDARY_LIST_DELIMITER}")))
      .map { case (arg, value) =>
        s"$arg${ScoptParserHelpers.KV_DELIMITER}$value"
      }
      .mkString(ScoptParserHelpers.LIST_DELIMITER)

    val expectedSeq = Seq(expected1, expected2, expected3, expected4)
    val resultPairs = ScoptParserHelpers.coordinateConfigsToStrings(coordinateConfigurations).zip(expectedSeq)

    resultPairs.foreach { case (actual, expected) =>
      assertEquals(actual, expected)
    }
  }
}
