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
package com.linkedin.photon.ml.test

import org.testng.{ITestResult, TestListenerAdapter}

/**
 * Implements a TestNG listener that converts "skip" status to "fail". This is a global approach to a problem where
 * exceptions thrown in the code for a DataProvider cause dependent tests to be silently skipped. The listener surfaces
 * any skip with an associated Throwable as a failure.
 */
class FailOnSkipListener extends TestListenerAdapter {

  /**
   * Invoked each time a test is skipped.
   *
   * @param tr ITestResult containing information about the run test
   */
  override def onTestSkipped(tr: ITestResult) {
    // If the skip was a result of an exception, change the skip to a failure
    if (Option(tr.getThrowable).isDefined) {
      tr.setStatus(ITestResult.FAILURE)
    }
  }
}
