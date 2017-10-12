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

import org.apache.spark.ml.param.ParamMap
import org.testng.Assert._
import org.testng.annotations.Test

/**
 * Unit tests for [[ScoptGameParametersParser]].
 */
class ScoptGameParametersParserTest {

  import ScoptGameParametersParserTest._

  /**
   * Test that (argument, value) pairs are formatted correctly for documentation.
   */
  @Test
  def testFormatArgs(): Unit = {

    val args = Seq("one", "two", "three")
    val vals = Seq(1, 2, 3)
    val expected = "one=1, two=2, three=3"

    assertEquals(MockScoptGameParametersParser.formatArgs(args.zip(vals.map(_.toString)).toMap), expected)
  }
}

object ScoptGameParametersParserTest {

  /**
   * Mock class to make the formatArgs function accessible.
   */
  object MockScoptGameParametersParser extends ScoptGameParametersParser {

    def parseFromCommandLine(args: Array[String]): ParamMap = ParamMap.empty

    def printForCommandLine(paramMap: ParamMap): Seq[String] = Seq()

    override def formatArgs(args: Map[String, String]): String = super.formatArgs(args)
  }
}
