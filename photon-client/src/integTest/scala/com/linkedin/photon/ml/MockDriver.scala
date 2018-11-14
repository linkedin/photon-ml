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
package com.linkedin.photon.ml

import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.testng.Assert._

import com.linkedin.photon.ml.stat.FeatureDataStatistics
import com.linkedin.photon.ml.util.PhotonLogger

/**
 * This is a mock Driver which extends from Photon-ML Driver. It's used to expose protected fields/methods for test
 * purpose.
 */
// TODO: Remove along with legacy Driver
class MockDriver(
    override val params: Params,
    override val sc: SparkContext,
    override val logger: PhotonLogger,
    override val seed: Long)
  extends Driver(params: Params, sc: SparkContext, logger: PhotonLogger, seed) {

  /**
   * Have the input features been summarized
   */
  private var isSummarized = false

  /**
   * Get the sequence of completed stages up to and including the current stage.
   *
   * @return An array of DriverStage objects
   */
  def stages: Array[DriverStage] = {
    (stageHistory += stage).toArray
  }

  /**
   * Get the validation metrics for the trained models.
   *
   * @return A map of (lambda -> map of (metric name -> metric value))
   */
  def metrics: Map[Double, Map[String, Double]] = perModelMetrics

  /**
   *
   * @param outputDir
   * @return
   */
  override def summarizeFeatures(outputDir: Option[Path]): FeatureDataStatistics = {
    isSummarized = true
    super.summarizeFeatures(outputDir)
  }
}

object MockDriver {

  // Use a static random seed for deterministic test results
  private val seed = 3L

  /**
   * Setup a mock Photon-ML Driver and run it, then verify that the actual results match the expected results.
   *
   * @param args The Driver runtime arguments
   * @param sc The Spark context
   * @param expectedStages The Photon-ML stages the mock run is expected to pass through
   * @param expectedNumFeatures The expected number of features in the input data
   * @param expectedNumTrainingData The expected number of training records
   * @param expectedIsSummarized Whether feature summarization was expected or not
   */
  def runLocally(
      args: Array[String],
      sc: SparkContext,
      expectedStages: Array[DriverStage],
      expectedNumFeatures: Int,
      expectedNumTrainingData: Int,
      expectedIsSummarized: Boolean): MockDriver = {

    // Parse the parameters from command line, should always be the 1st line in main
    val params = PhotonMLCmdLineParser.parseFromCommandLine(args)
    val logPath = new Path(params.outputDir, "log-message.txt")
    val logger = new PhotonLogger(logPath, sc)
    val job = new MockDriver(params, sc, logger, seed)
    job.run()

    val actualStages = job.stages

    assertEquals(actualStages, expectedStages,
      "The actual stages Driver went through, " + actualStages.mkString(",") +
          " is inconsistent with the expected one")
    assertEquals(job.numFeatures(), expectedNumFeatures,
      "The number of features " + job.numFeatures() + " do not meet the expectation.")
    assertEquals(job.numTrainingData(), expectedNumTrainingData,
      "The number of training data points " + job.numTrainingData() + " do not meet the expectation")
    assertEquals(job.isSummarized, expectedIsSummarized)

    // Closing up
    logger.close()

    job
  }
}
