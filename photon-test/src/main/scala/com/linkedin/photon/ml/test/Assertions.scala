/*
 * Copyright 2015 LinkedIn Corp. All rights reserved.
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
package com.linkedin.photon.ml.test

import org.testng.Assert.assertEquals


/**
 * This object provides utility for complex assertions.
 * @author dpeng
 */
object Assertions {

  /**
   * This method compares lists of doubles with a tolerance.
   * @param actual The actual iterable
   * @param expected The expected iterable
   * @param delta The tolerance
   */
  def assertIterableEqualsWithTolerance(actual: Iterable[Double], expected: Iterable[Double], delta: Double): Unit = {
    actual.zip(expected).foreach {
      case (act, exp) => assertEquals(act, exp, delta)
    }
  }
}
