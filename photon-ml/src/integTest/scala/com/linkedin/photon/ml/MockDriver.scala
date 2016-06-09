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
package com.linkedin.photon.ml

import com.linkedin.photon.ml.stat.BasicStatisticalSummary
import com.linkedin.photon.ml.util.PhotonLogger
import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.testng.Assert._

/**
 * This is a mock Driver which extends from Photon-ML Driver. It's used to expose protected fields/methods for test
 * purpose.
 *
 * @author yizhou
 * @author dpeng
 */
class MockDriver(
    override val params: Params,
    override val sc: SparkContext,
    override val logger: PhotonLogger,
    override val seed: Long)
  extends Driver(params: Params, sc: SparkContext, logger: PhotonLogger, seed) {

  var isSummarized = false

  def stages(): Array[DriverStage] = {
    (stageHistory += stage).toArray
  }

  override def summarizeFeatures(outputDir: Option[String]): BasicStatisticalSummary = {
    isSummarized = true
    super.summarizeFeatures(outputDir)
  }
}


object MockDriver {

  // Use a static random seed for deterministic test results
  val seed = 3L

  def runLocally(
      args: Array[String],
      sparkContext: SparkContext,
      expectedStages: Array[DriverStage],
      expectedNumFeatures: Int,
      expectedNumTrainingData: Int,
      expectedIsSummarized: Boolean): Unit = {

    /* Parse the parameters from command line, should always be the 1st line in main*/
    val params = PhotonMLCmdLineParser.parseFromCommandLine(args)
    val logPath = new Path(params.outputDir, "log-message.txt")
    val logger = new PhotonLogger(logPath, sparkContext)
    val job = new MockDriver(params, sparkContext, logger, seed)
    job.run()

    val actualStages = job.stages()

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
  }
}
