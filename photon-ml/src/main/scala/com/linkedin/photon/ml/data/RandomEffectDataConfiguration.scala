package com.linkedin.photon.ml.data

import com.linkedin.photon.ml.projector.{IdentityProjection, IndexMapProjection, RandomProjection, ProjectorType}
import com.linkedin.photon.ml.projector.ProjectorType._

/**
 * Configurations needed in order to generate a [[RandomEffectDataSet]]
 *
 * @param randomEffectId The corresponding random effect Id of the data set
 * @param featureShardId Key of the feature shard used to generate the data set
 * @param numActiveDataPointsToKeepUpperBound The upper bound on the number of samples to keep (via reservoir sampling)
 *                                           as "active" for each individual-id level local data set in the random
 *                                           effect data set. The remaining samples will be kept as "passive" data.
 * @param numPassiveDataPointsToKeepLowerBound The lower bound on the number of passive data points to keep for each
 *                                           individual-id level local data set. Only those individual-id level local
 *                                           data set with number of passive data points larger than
 *                                           [[numPassiveDataPointsToKeepLowerBound]] will have its passive data kept
 *                                           during the processing step, and as a result all the remaining passive data
 *                                           have more than [[numPassiveDataPointsToKeepLowerBound]] samples.
 * @param numFeaturesToSamplesRatioUpperBound The upper bound on the ratio between number of features and number of
 *                                            samples used for feature selection for each individual-id level local
 *                                            data set in the random effect data set.
 */
case class RandomEffectDataConfiguration(
    randomEffectId: String,
    featureShardId: String,
    numPartitions: Int,
    numActiveDataPointsToKeepUpperBound: Int = Int.MaxValue,
    numPassiveDataPointsToKeepLowerBound: Int = 0,
    numFeaturesToSamplesRatioUpperBound: Double = Double.MaxValue,
    projectorType: ProjectorType) {

  def isDownSamplingNeeded = numActiveDataPointsToKeepUpperBound < Int.MaxValue

  def isFeatureSelectionNeeded = numFeaturesToSamplesRatioUpperBound < Double.MaxValue

  override def toString: String = {
    s"randomEffectId: $randomEffectId, featureShardId: $featureShardId, numPartitions: $numPartitions, " +
        s"numActiveDataPointsToKeepUpperBound: $numActiveDataPointsToKeepUpperBound, " +
        s"numPassiveDataPointsToKeepLowerBound: $numPassiveDataPointsToKeepLowerBound, " +
        s"numFeaturesToSamplesRatioUpperBound: $numFeaturesToSamplesRatioUpperBound, " +
        s"projectorType: $projectorType."
  }
}

object RandomEffectDataConfiguration {

  private val FIRST_LEVEL_SPLITTER = ","
  private val SECOND_LEVEL_SPLITTER = "="

  //TODO: Need a better way to parse the configuration from a structured string, or better from a text/JSON file
  /**
   * Parse and build the [[RandomEffectDataConfiguration]] from the input [[String]]
   * @param string The input [[String]]
   * @return The parsed and built random effect data configuration
   */
  def parseAndBuildFromString(string: String): RandomEffectDataConfiguration = {

    val expectedTokenLength = 7
    val configParams = string.split(FIRST_LEVEL_SPLITTER)
    assert(configParams.length == expectedTokenLength, s"Cannot parse $string as random effect data configuration.\n" +
        s"The expected random effect data configuration should contain $expectedTokenLength parts separated by " +
        s"\'$FIRST_LEVEL_SPLITTER\'.")

    val randomEffectId = configParams(0)
    val featureShardKey = configParams(1)
    val numPartitions = configParams(2).toInt
    val rawUpperBoundNumActiveDataPointsToKeep = configParams(3).toInt
    val upperBoundNumActiveDataPointsToKeep = if (rawUpperBoundNumActiveDataPointsToKeep < 0) {
      Int.MaxValue
    } else {
      rawUpperBoundNumActiveDataPointsToKeep
    }
    val rawLowerBoundNumPassiveDataPointsToKeep = configParams(4).toInt
    val lowerBoundNumPassiveDataPointsToKeep = if (rawLowerBoundNumPassiveDataPointsToKeep < 0) {
      0
    } else {
      rawLowerBoundNumPassiveDataPointsToKeep
    }
    val rawUpperBoundNumFeaturesToSamplesRatio = configParams(5).toDouble
    val upperBoundNumFeaturesToSamplesRatio = if (rawUpperBoundNumFeaturesToSamplesRatio < 0) {
      Double.MaxValue
    } else {
      rawUpperBoundNumFeaturesToSamplesRatio
    }

    val projectorConfigParams = configParams(6).split(SECOND_LEVEL_SPLITTER)
    val projectorTypeName = ProjectorType.withName(projectorConfigParams.head.toUpperCase)
    val projectorType = projectorTypeName match {
      case RANDOM =>
        assert(projectorConfigParams.length == 2, s"If projector of type $RANDOM is selected, the projected space " +
            s"dimension needs to be specified. Correct configuration format is " +
            s"$RANDOM${SECOND_LEVEL_SPLITTER}projectedSpaceDimension.")
        val projectedSpaceDimension = projectorConfigParams.last.toInt
        RandomProjection(projectedSpaceDimension)
      case INDEX_MAP => IndexMapProjection
      case IDENTITY => IdentityProjection
      case _ => throw new UnsupportedOperationException(s"Unsupported projector name $projectorTypeName")
    }

    RandomEffectDataConfiguration(randomEffectId, featureShardKey, numPartitions, upperBoundNumActiveDataPointsToKeep,
      lowerBoundNumPassiveDataPointsToKeep, upperBoundNumFeaturesToSamplesRatio, projectorType)
  }
}
