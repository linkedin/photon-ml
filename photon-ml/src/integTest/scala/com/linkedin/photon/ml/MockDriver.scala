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

import com.linkedin.photon.ml.io.LogWriter
import org.apache.spark.{SparkConf, SparkContext}
import org.testng.Assert._

/**
 * This is a mock Driver which extends from Photon-ML Driver. It's used to expose protected fields/methods for test
 * purpose.
 *
 * @author yizhou
 * @author dpeng
 */
class MockDriver(override val params: Params, override val sc: SparkContext, override val logger: LogWriter)
    extends Driver(params: Params, sc: SparkContext, logger: LogWriter) {
  var isSummarized = false

  def stages(): Array[DriverStage] = {
    (stageHistory += stage).toArray
  }

  override def summarizeFeatures(outputDir: Option[String]) = {
    isSummarized = true
    super.summarizeFeatures(outputDir)
  }
}


object MockDriver {

  def runLocally(args: Array[String], expectedStages: Array[DriverStage], expectedNumFeatures: Int,
      expectedNumTrainingData: Int, expectedIsSummarized: Boolean): Unit = {
    /* Parse the parameters from command line, should always be the 1st line in main*/
    val params = PhotonMLCmdLineParser.parseFromCommandLine(args)
    /* Configure the Spark application and initialize SparkContext, which is the entry point of a Spark application */
    val sc: SparkContext = SparkContextConfiguration.asYarnClient(new SparkConf().setMaster("local[4]"),
                                                                  params.jobName,
                                                                  params.kryo)
    try {
    val logger = new LogWriter(params.outputDir, sc)
      val job = new MockDriver(params, sc, logger)
      job.run()

      val actualStages = job.stages()

      assertEquals(actualStages, expectedStages,
        "The actual stages Driver went through is inconsistent with the expected one.")
      assertEquals(job.numFeatures(), expectedNumFeatures,
        "The number of features do not meet the expectation.")
      assertEquals(job.numTrainingData(), expectedNumTrainingData,
        "The number of training data points do not meet the expectation")
      assertEquals(job.isSummarized, expectedIsSummarized)

      // Closing up
      logger.close()
    } finally {
      // Make sure sc is stopped
      sc.stop()
    }
  }
}
