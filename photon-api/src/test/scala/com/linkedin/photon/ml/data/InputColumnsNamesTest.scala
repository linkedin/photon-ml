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
import org.testng.annotations.{DataProvider, Test}

/**
 * Unit tests for [[InputColumnsNames]]
 */
class InputColumnsNamesTest {

  /**
   * Test that the default column names match the enum values.
   */
  @Test
  def testDefaultConstructor(): Unit = {

    val columnsNames = InputColumnsNames()

    InputColumnsNames.all.foreach(n => assertEquals(columnsNames(n), n.toString))
  }

  @DataProvider
  def validNames(): Array[Array[Object]] = {
    Array(
      Array(Map(InputColumnsNames.UID -> "newUID")),
      Array(Map(InputColumnsNames.RESPONSE -> "newResponse")),
      Array(Map(InputColumnsNames.OFFSET -> "newOffset")),
      Array(Map(InputColumnsNames.WEIGHT -> "newWeight")),
      Array(Map(InputColumnsNames.META_DATA_MAP -> "newMap")),
      Array(
        Map(
          InputColumnsNames.RESPONSE -> "newResponse",
          InputColumnsNames.OFFSET -> "newOffset",
          InputColumnsNames.WEIGHT -> "newWeight")))
  }

  /**
   * Test that some or all of the column names can be overwritten.
   */
  @Test(dataProvider = "validNames")
  def testConstructor(customNames: Map[InputColumnsNames.Value, String]): Unit = {

    val columnNames = InputColumnsNames(customNames)

    InputColumnsNames.all.foreach { column =>
      if (customNames.contains(column)) {
        assertEquals(columnNames(column), customNames(column))
      } else {
        assertEquals(columnNames(column), column.toString)
      }
    }
  }

  @DataProvider
  def invalidNames(): Array[Array[Object]] = {
    Array(
      Array(Map(InputColumnsNames.RESPONSE -> "uid")),
      Array(
        Map(
          InputColumnsNames.RESPONSE -> "sameName",
          InputColumnsNames.OFFSET -> "sameName")))
  }

  /**
   * Test that multiple required columns with the same name will be rejected.
   */
  @Test(dataProvider = "invalidNames", expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testInvalidInput(customNames: Map[InputColumnsNames.Value, String]): Unit = InputColumnsNames(customNames)
}
