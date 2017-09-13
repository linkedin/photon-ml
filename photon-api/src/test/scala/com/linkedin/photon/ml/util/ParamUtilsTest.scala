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
package com.linkedin.photon.ml.util

import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.Identifiable
import org.mockito.Mockito._
import org.testng.Assert.{assertEquals, assertFalse, assertTrue}
import org.testng.annotations.Test

/**
 * Tests for [[ParamUtils]].
 */
class ParamUtilsTest {

  import ParamUtilsTest._

  /**
   * Test that [[Param]] objects are correctly created.
   */
  @Test
  def testParamCreation(): Unit = {

    implicit val mockIdentifiable = mock(classOf[Identifiable])
    val mockUid = "uid"
    val mockName = "name"
    val mockDoc = "doc"
    val mockValue = "value"

    doReturn(mockUid).when(mockIdentifiable).uid

    val param1: Param[Any] = ParamUtils.createParam(mockName, mockDoc)
    val param2: Param[Any] = ParamUtils.createParam(mockName, mockDoc, alwaysFalse)

    assertTrue(param1.isValid(mockValue))
    assertFalse(param2.isValid(mockValue))
    assertEquals(param1.parent, param2.parent)
  }
}

object ParamUtilsTest {

  /**
   * Validator function that returns false for all input.
   *
   * @param x Some input
   * @return False
   */
  private def alwaysFalse(x: Any): Boolean = false
}
