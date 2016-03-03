package com.linkedin.photon.ml.data

/**
 * Configuration for a fixed effect dataset
 *
 * @author xazhang
 */
case class FixedEffectDataConfiguration(featureShardId: String, numPartitions: Int) {
  override def toString: String = s"featureShardId: $featureShardId, numPartitions: $numPartitions"
}

object FixedEffectDataConfiguration {

  private val SPLITTER = ","

  /**
   * Parse and build the configuration object from a string representation
   *
   * @param string the string representation
   * @return the configuration object
   */
  def parseAndBuildFromString(string: String): FixedEffectDataConfiguration = {

    val expectedTokenLength = 2
    val configParams = string.split(SPLITTER)
    assert(configParams.length == expectedTokenLength, s"Cannot parse $string as fixed effect data configuration.\n" +
        s"The expected fixed effect data configuration should contain $expectedTokenLength parts separated by " +
        s"\'$SPLITTER\'.")

    val featureShardId = configParams(0)
    val numPartitions = configParams(1).toInt
    FixedEffectDataConfiguration(featureShardId, numPartitions)
  }
}
