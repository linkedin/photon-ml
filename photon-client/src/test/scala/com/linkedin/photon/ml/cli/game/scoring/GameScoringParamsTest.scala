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
package com.linkedin.photon.ml.cli.game.scoring

import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.test.CommonTestUtils._

/**
 * Unit tests for [[GameScoringParams]].
 */
class GameScoringParamsTest {

  import GameScoringParamsTest._

  @DataProvider
  def requiredOptions(): Array[Array[Any]] = {
    REQUIRED_OPTIONS.map(optionName => Array[Any](optionName))
  }

  /**
   * Test that parameter parsing will fail if one of the required arguments is not present.
   *
   * @param optionName The required argument name
   */
  @Test(dataProvider = "requiredOptions", expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testMissingRequiredArg(optionName: String): Unit =
    GameScoringParams.parseFromCommandLine(mapToArray(requiredArgsMissingOne(optionName)))
}

object GameScoringParamsTest {

  // Required parameters
  private val INPUT_DATA_DIRS = "input-data-dirs"
  private val GAME_MODEL_INPUT_DIR = "game-model-input-dir"
  private val OUTPUT_DIR = "output-dir"
  private val FEATURE_NAME_AND_TERM_SET_PATH = "feature-name-and-term-set-path"

  val REQUIRED_OPTIONS = Array(INPUT_DATA_DIRS, GAME_MODEL_INPUT_DIR, OUTPUT_DIR, FEATURE_NAME_AND_TERM_SET_PATH)

  /**
   * Get all required arguments except the one with name missingArgName.
   *
   * @param missingOptionName The name of the argument to exclude
   * @return A map of required arguments, except for one
   */
  def requiredArgsMissingOne(missingOptionName: String): Map[String, String] = {
    if (REQUIRED_OPTIONS.isEmpty) {
      throw new RuntimeException("No required option configured in test.")
    }
    val args = new Array[(String, String)](REQUIRED_OPTIONS.length - 1)
    var i = 0
    REQUIRED_OPTIONS.filter(_ != missingOptionName).foreach { option =>
      val name = fromOptionNameToArg(option)
      args(i) = (name, option)
      i += 1
    }
    args.toMap
  }
}
