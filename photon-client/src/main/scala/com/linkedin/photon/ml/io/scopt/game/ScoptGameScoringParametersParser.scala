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

import scala.language.existentials

import org.apache.hadoop.fs.Path
import org.apache.spark.ml.param.ParamMap
import scopt.{OptionDef, OptionParser, Read}

import com.linkedin.photon.ml.cli.game.scoring.GameScoringDriver
import com.linkedin.photon.ml.io.scopt.{ScoptParameter, ScoptParserReads}

/**
 * Scopt command line argument parser for GAME scoring parameters.
 */
object ScoptGameScoringParametersParser extends ScoptGameParametersParser {

  import ScoptParserReads._

  val scoptGameScoringParams: Seq[ScoptParameter[In, Out] forSome { type In; type Out }] =
    createScoptGameParams(GameScoringDriver) ++ Seq(

      // Model Input Dir
      ScoptParameter[Path, Path](
        GameScoringDriver.modelInputDirectory,
        usageText = "<path>",
        isRequired = true),

      // Model ID
      ScoptParameter[String, String](
        GameScoringDriver.modelId,
        usageText = "<value>"),

      // Log Model Data and Stats
      ScoptParameter[Boolean, Boolean](
        GameScoringDriver.logDataAndModelStats),

      // Spill Data to Disk
      ScoptParameter[Boolean, Boolean](
        GameScoringDriver.spillScoresToDisk))

  override protected val parser: OptionParser[ParamMap] = new OptionParser[ParamMap]("GAME-Scoring") {

    /**
     * Helper method to convert a [[ScoptParameter]] object to a defined Scopt parameter object.
     *
     * @tparam In The type of the command line argument when parsed by Scopt
     * @param scoptParameter A Scopt wrapper for a [[org.apache.spark.ml.param.Param]] which contains extra information
     *                       to define how to parse/print it from/to the command line
     * @return A Scopt [[OptionDef]]
     */
    private def optHelper[In](scoptParameter: ScoptParameter[In, _]): OptionDef[In, ParamMap] = {

      implicit val read: Read[In] = scoptParameter.read

      scoptParameter.toOptionDef(opt[In])
    }

    scoptGameScoringParams.foreach { optHelper(_) }
  }

  /**
   * Convert parameters stored in a valid [[ParamMap]] object to [[String]] format for output to the command line, in a
   * format which can be parsed back into a valid [[ParamMap]].
   *
   * @param paramMap A valid [[ParamMap]]
   * @return A [[Seq]] of [[String]] representations of the parameters, in a format that can be parsed by Scopt
   */
  override def printForCommandLine(paramMap: ParamMap): Seq[String] = {

    GameScoringDriver.validateParams(paramMap)

    scoptGameScoringParams.flatMap(_.generateCmdLineArgs(paramMap))
  }
}
