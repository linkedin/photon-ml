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

import scala.language.existentials

import org.apache.hadoop.fs.Path
import org.apache.spark.ml.param.ParamMap
import scopt.{OptionDef, OptionParser, Read}

import com.linkedin.photon.ml.Types.REType
import com.linkedin.photon.ml.cli.game.scoring.GameScoringDriver

/**
 * Scopt command line argument parser for GAME scoring parameters.
 */
object ScoptGameScoringParametersParser extends ScoptGameParametersParser {

  import ScoptParserReads._

  val scoptGameScoringParams: Seq[ScoptParameter[In, Out] forSome { type In; type Out }] =
    createScoptGameParams(GameScoringDriver) ++ Seq(

      // Random effect types
      ScoptParameter[Seq[REType], Set[REType]](
        GameScoringDriver.randomEffectTypes,
        parse = ScoptParserHelpers.parseSetFromSeq,
        print = ScoptParserHelpers.iterableToString,
        usageText = "<path1>,<path2>,...",
        isRequired = true),

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

  /**
   * Parse command line arguments for GAME scoring into a [[ParamMap]].
   *
   * @param args [[Array]] of command line arguments
   * @return An initialized [[ParamMap]]
   */
  def parseFromCommandLine(args: Array[String]): ParamMap = {

    val parser = new OptionParser[ParamMap]("GAME-Scoring") {

      private def optHelper[In](scoptParameter: ScoptParameter[In, _]): OptionDef[In, ParamMap] = {

        implicit val read: Read[In] = scoptParameter.read

        scoptParameter.toOptionDef(opt[In])
      }

      scoptGameScoringParams.foreach { optHelper(_) }
    }

    parser.parse(args, ParamMap.empty) match {
      case Some(params) => params

      case None =>
        val errMsg = new StringBuilder()

        for (i <- args.indices by 2) {
          errMsg.append(s"${args(i)} ${args(i + 1)}\n")
        }

        throw new IllegalArgumentException(s"Parsing the following command line arguments failed:\n${errMsg.toString()}")
    }
  }

  /**
   * Given a [[ParamMap]] of valid parameters, convert them into a [[Seq]] of [[String]] representations which can be
   * parsed by Scopt.
   *
   * @param paramMap Valid GAME scoring parameters
   * @return A [[Seq]] of [[String]] representations of the parameters, in a format that can be parsed by Scopt
   */
  def printForCommandLine(paramMap: ParamMap): Seq[String] = {

    GameScoringDriver.validateParams(paramMap)

    scoptGameScoringParams.flatMap(_.generateCmdLineArgs(paramMap))
  }
}
