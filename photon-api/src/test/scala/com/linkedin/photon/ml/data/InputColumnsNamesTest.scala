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
package com.linkedin.photon.ml.data

import org.testng.Assert._
import org.testng.annotations.Test

/**
 * Unit tests for InputColumnsNames
 */
class InputColumnsNamesTest {

  @Test
  def testDefaultConstructor(): Unit = {

    val columnsNames = InputColumnsNames()
    InputColumnsNames.all.foreach(n => assertEquals(columnsNames(n), n.toString))

    val defaultStr = "uid: uid, response: response, offset: offset, weight: weight, metadataMap: metadataMap"
    assertEquals(columnsNames.toString, defaultStr)
  }

  @Test
  def testAccessors(): Unit = {

    val columnsNames = InputColumnsNames()
    columnsNames.updated(InputColumnsNames.RESPONSE, "label")
    assertEquals(columnsNames(InputColumnsNames.RESPONSE), "label")
  }
}
