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
package com.linkedin.photon.ml.io.scopt.game

import org.apache.hadoop.fs.Path
import org.apache.spark.ml.param.ParamMap
import org.testng.Assert.{assertEquals, assertTrue}
import org.testng.annotations.Test

import com.linkedin.photon.ml.cli.game.training.GameTrainingDriver
import com.linkedin.photon.ml.data.{FixedEffectDataConfiguration, InputColumnsNames, RandomEffectDataConfiguration}
import com.linkedin.photon.ml.evaluation.EvaluatorType.{AUC, RMSE}
import com.linkedin.photon.ml.io.{FeatureShardConfiguration, FixedEffectCoordinateConfiguration, ModelOutputMode, RandomEffectCoordinateConfiguration}
import com.linkedin.photon.ml.normalization.NormalizationType
import com.linkedin.photon.ml.optimization.game.{FixedEffectOptimizationConfiguration, RandomEffectOptimizationConfiguration}
import com.linkedin.photon.ml.optimization._
import com.linkedin.photon.ml.projector.IndexMapProjection
import com.linkedin.photon.ml.util.{DateRange, DoubleRange, PhotonLogger}
import com.linkedin.photon.ml.{DataValidationType, HyperparameterTunerName, HyperparameterTuningMode, TaskType}

/**
 * Unit tests for the [[ScoptGameTrainingParametersParser]].
 */
class ScoptGameTrainingParametersParserTest {

  /**
   * Test that a valid [[ParamMap]] can be roundtrip-ed by the parser (parameters -> string args -> parameters).
   */
  @Test
  def testRoundTrip(): Unit = {

    val inputPaths = Set(new Path("/some/input/path"))
    val inputDateRange = DateRange.fromDateString("20170101-20181231")
    val offHeapIndexMapPath = new Path("/some/off/heap/path")
    val offHeapIndexMapPartitions = 1
    val customColumnsNames = InputColumnsNames(InputColumnsNames.all.map(col => (col, s"___${col.toString}")).toMap)
    val evaluators = Seq(AUC, RMSE)
    val outputPath = new Path("/some/output/path")
    val overrideOutputDir = true
    val dataValidation = DataValidationType.VALIDATE_SAMPLE
    val logLevel = PhotonLogger.parseLogLevelString("WARN")
    val applicationName = "myApplication_name"
    val trainingTask = TaskType.POISSON_REGRESSION
    val validationPaths = Set(new Path("/some/validation/path"))
    val validationDateRange = DateRange.fromDateString("20160101-20171231")
    val validationPartitions = 2
    val partialRetrainModelDir = new Path("/some/existing/model/path")
    val outputMode = ModelOutputMode.EXPLICIT
    val outputFilesLimit = 3
    val coordinateDescentIter = 4
    val normalization = NormalizationType.SCALE_WITH_MAX_MAGNITUDE
    val dataSummaryPath = new Path("/some/summary/path")
    val treeAggregateDepth = 5
    val hyperparameterTunerName = HyperparameterTunerName.DUMMY
    val hyperparameterTuningMode = HyperparameterTuningMode.BAYESIAN
    val hyperparameterTuningIter = 6
    val varianceComputation = VarianceComputationType.SIMPLE
    val ignoreThreshold = true

    val featureShard1 = "featureShard1"
    val featureBags1 = Set("bag1", "bag2")
    val featureShardIntercept1 = true
    val featureShardConfig1 = FeatureShardConfiguration(featureBags1, featureShardIntercept1)
    val featureShard2 = "featureShard2"
    val featureBags2 = Set("bag3", "bag4")
    val featureShardIntercept2 = false
    val featureShardConfig2 = FeatureShardConfiguration(featureBags2, featureShardIntercept2)
    val featureShardConfigs = Map(
      (featureShard1, featureShardConfig1),
      (featureShard2, featureShardConfig2))

    val fixedEffectCoordinateId = "fixedCoordinate"
    val fixedEffectPartitions = 7
    val fixedEffectDataConfiguration = FixedEffectDataConfiguration(featureShard1, fixedEffectPartitions)
    val fixedEffectOptimizerType = OptimizerType.LBFGS
    val fixedEffectMaxIter = 8
    val fixedEffectTolerance = 9.0
    val fixedEffectOptimizerConfig = OptimizerConfig(
      fixedEffectOptimizerType,
      fixedEffectMaxIter,
      fixedEffectTolerance)
    val fixedEffectDownSamplingRate = 0.1
    val fixedEffectOptimizationConfiguration = FixedEffectOptimizationConfiguration(
      fixedEffectOptimizerConfig,
      downSamplingRate = fixedEffectDownSamplingRate)
    val fixedEffectCoordinateConfiguration = FixedEffectCoordinateConfiguration(
      fixedEffectDataConfiguration,
      fixedEffectOptimizationConfiguration)
    val randomEffectCoordinateId = "randomCoordinate"
    val randomEffectType = "myType"
    val randomEffectPartitions = 10
    val randomEffectActiveLowerBound = Some(11)
    val randomEffectActiveUpperBound = Some(11)
    val randomEffectPassiveLowerBound = Some(12)
    val randomEffectFeatureRatio = Some(13.0)
    val randomEffectDataConfiguration = RandomEffectDataConfiguration(
      randomEffectType,
      featureShard2,
      randomEffectPartitions,
      randomEffectActiveLowerBound,
      randomEffectActiveUpperBound,
      randomEffectPassiveLowerBound,
      randomEffectFeatureRatio,
      IndexMapProjection)
    val randomEffectOptimizerType = OptimizerType.TRON
    val randomEffectMaxIter = 14
    val randomEffectTolerance = 15.0
    val randomEffectOptimizerConfig = OptimizerConfig(
      randomEffectOptimizerType,
      randomEffectMaxIter,
      randomEffectTolerance)
    val randomEffectRegularizationContext = L1RegularizationContext
    val randomEffectRegularizationWeightRange = DoubleRange(1.0, 100.0)
    val randomEffectElasticNetParamRange = DoubleRange(0.0, 0.5)
    val randomEffectOptimizationConfiguration = RandomEffectOptimizationConfiguration(
      randomEffectOptimizerConfig,
      randomEffectRegularizationContext,
      regularizationWeightRange = Some(randomEffectRegularizationWeightRange),
      elasticNetParamRange = Some(randomEffectElasticNetParamRange))
    val randomEffectRegularizationWeights = Set(16.0, 17.0)
    val randomEffectCoordinateConfiguration = RandomEffectCoordinateConfiguration(
      randomEffectDataConfiguration,
      randomEffectOptimizationConfiguration,
      randomEffectRegularizationWeights)
    val coordinateConfigs = Map(
      (fixedEffectCoordinateId, fixedEffectCoordinateConfiguration),
      (randomEffectCoordinateId, randomEffectCoordinateConfiguration))
    val lockedCoordinateId = "lockedCoordinate"
    val partialRetrainLockedCoordinates = Set(lockedCoordinateId)
    val coordinateUpdateSequence = Seq(fixedEffectCoordinateId, randomEffectCoordinateId, lockedCoordinateId)

    val initialParamMap = ParamMap
      .empty
      .put(GameTrainingDriver.inputDataDirectories, inputPaths)
      .put(GameTrainingDriver.inputDataDateRange, inputDateRange)
      .put(GameTrainingDriver.offHeapIndexMapDirectory, offHeapIndexMapPath)
      .put(GameTrainingDriver.offHeapIndexMapPartitions, offHeapIndexMapPartitions)
      .put(GameTrainingDriver.inputColumnNames, customColumnsNames)
      .put(GameTrainingDriver.evaluators, evaluators)
      .put(GameTrainingDriver.rootOutputDirectory, outputPath)
      .put(GameTrainingDriver.overrideOutputDirectory, overrideOutputDir)
      .put(GameTrainingDriver.outputFilesLimit, outputFilesLimit)
      .put(GameTrainingDriver.featureShardConfigurations, featureShardConfigs)
      .put(GameTrainingDriver.dataValidation, dataValidation)
      .put(GameTrainingDriver.logLevel, logLevel)
      .put(GameTrainingDriver.applicationName, applicationName)
      .put(GameTrainingDriver.trainingTask, trainingTask)
      .put(GameTrainingDriver.validationDataDirectories, validationPaths)
      .put(GameTrainingDriver.validationDataDateRange, validationDateRange)
      .put(GameTrainingDriver.minValidationPartitions, validationPartitions)
      .put(GameTrainingDriver.modelInputDirectory, partialRetrainModelDir)
      .put(GameTrainingDriver.partialRetrainLockedCoordinates, partialRetrainLockedCoordinates)
      .put(GameTrainingDriver.outputMode, outputMode)
      .put(GameTrainingDriver.coordinateDescentIterations, coordinateDescentIter)
      .put(GameTrainingDriver.coordinateConfigurations, coordinateConfigs)
      .put(GameTrainingDriver.coordinateUpdateSequence, coordinateUpdateSequence)
      .put(GameTrainingDriver.normalization, normalization)
      .put(GameTrainingDriver.dataSummaryDirectory, dataSummaryPath)
      .put(GameTrainingDriver.treeAggregateDepth, treeAggregateDepth)
      .put(GameTrainingDriver.hyperParameterTunerName, hyperparameterTunerName)
      .put(GameTrainingDriver.hyperParameterTuning, hyperparameterTuningMode)
      .put(GameTrainingDriver.hyperParameterTuningIter, hyperparameterTuningIter)
      .put(GameTrainingDriver.varianceComputationType, varianceComputation)
      .put(GameTrainingDriver.ignoreThresholdForNewModels, ignoreThreshold)

    val finalParamMap = ScoptGameTrainingParametersParser.parseFromCommandLine(
      ScoptGameTrainingParametersParser.printForCommandLine(initialParamMap).flatMap(_.split(" ")).toArray)

    ScoptGameTrainingParametersParser
      .scoptGameTrainingParams
      .filterNot(_.param.eq(GameTrainingDriver.coordinateConfigurations))
      .foreach { scoptParam =>
        assertEquals(finalParamMap.get(scoptParam.param), initialParamMap.get(scoptParam.param))
      }

    //
    // Check coordinate configurations separately. This is done as an alternative to custom hashCode() and equals()
    // implementations for CoordinateConfiguration, CoordinateDataConfiguration, CoordinateOptimizationConfiguration,
    // and OptimizerConfig classes (which are otherwise never compared directly against each other).
    //

    val finalCoordinateConfigs = finalParamMap.get(GameTrainingDriver.coordinateConfigurations).get

    // Compare configurations map
    assertEquals(finalCoordinateConfigs.size, coordinateConfigs.size)
    assertTrue(coordinateConfigs.keySet.forall(finalCoordinateConfigs.contains))

    val finalFixedCoordinateConfig =
      finalCoordinateConfigs(fixedEffectCoordinateId).asInstanceOf[FixedEffectCoordinateConfiguration]
    val finalFixedDataCoordinateConfig = finalFixedCoordinateConfig.dataConfiguration
    val finalFixedOptimizationCoordinateConfig = finalFixedCoordinateConfig.optimizationConfiguration
    val finalFixedOptimizationConfig = finalFixedOptimizationCoordinateConfig.optimizerConfig
    val finalFixedRegularizationWeights = finalFixedCoordinateConfig.regularizationWeights
    val finalRandomCoordinateConfig =
      finalCoordinateConfigs(randomEffectCoordinateId).asInstanceOf[RandomEffectCoordinateConfiguration]
    val finalRandomDataCoordinateConfig = finalRandomCoordinateConfig.dataConfiguration
    val finalRandomOptimizationCoordinateConfig = finalRandomCoordinateConfig.optimizationConfiguration
    val finalRandomOptimizationConfig = finalRandomOptimizationCoordinateConfig.optimizerConfig
    val finalRandomRegularizationWeights = finalRandomCoordinateConfig.regularizationWeights

    // Compare fixed effect configuration
    assertEquals(finalFixedDataCoordinateConfig.featureShardId, featureShard1)
    assertEquals(finalFixedDataCoordinateConfig.minNumPartitions, fixedEffectPartitions)

    assertEquals(finalFixedOptimizationConfig.optimizerType, fixedEffectOptimizerType)
    assertEquals(finalFixedOptimizationConfig.tolerance, fixedEffectTolerance)
    assertEquals(finalFixedOptimizationConfig.maximumIterations, fixedEffectMaxIter)
    assertEquals(finalFixedOptimizationCoordinateConfig.regularizationContext, NoRegularizationContext)
    assertEquals(finalFixedOptimizationCoordinateConfig.regularizationWeight, 0D)
    assertEquals(finalFixedOptimizationCoordinateConfig.downSamplingRate, fixedEffectDownSamplingRate)

    assertEquals(finalFixedRegularizationWeights, Set(0D))

    // Compare random effect configuration
    assertEquals(finalRandomDataCoordinateConfig.randomEffectType, randomEffectType)
    assertEquals(finalRandomDataCoordinateConfig.featureShardId, featureShard2)
    assertEquals(finalRandomDataCoordinateConfig.minNumPartitions, randomEffectPartitions)
    assertEquals(finalRandomDataCoordinateConfig.numActiveDataPointsLowerBound, randomEffectActiveLowerBound)
    assertEquals(finalRandomDataCoordinateConfig.numActiveDataPointsUpperBound, randomEffectActiveUpperBound)
    assertEquals(finalRandomDataCoordinateConfig.numPassiveDataPointsLowerBound, randomEffectPassiveLowerBound)
    assertEquals(finalRandomDataCoordinateConfig.numFeaturesToSamplesRatioUpperBound, randomEffectFeatureRatio)
    assertEquals(finalRandomDataCoordinateConfig.projectorType, IndexMapProjection)

    assertEquals(finalRandomOptimizationConfig.optimizerType, randomEffectOptimizerType)
    assertEquals(finalRandomOptimizationConfig.tolerance, randomEffectTolerance)
    assertEquals(finalRandomOptimizationConfig.maximumIterations, randomEffectMaxIter)
    assertEquals(finalRandomOptimizationCoordinateConfig.regularizationContext, randomEffectRegularizationContext)
    assertEquals(finalRandomOptimizationCoordinateConfig.regularizationWeight, 0D)
    assertEquals(
      finalRandomOptimizationCoordinateConfig.regularizationWeightRange,
      Some(randomEffectRegularizationWeightRange))
    assertEquals(finalRandomOptimizationCoordinateConfig.elasticNetParamRange, Some(randomEffectElasticNetParamRange))

    assertEquals(finalRandomRegularizationWeights, randomEffectRegularizationWeights)
  }
}
