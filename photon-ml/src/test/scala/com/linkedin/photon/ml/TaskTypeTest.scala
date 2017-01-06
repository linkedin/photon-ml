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

import org.testng.Assert._
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.photon.ml.model.Coefficients
import com.linkedin.photon.ml.supervised.classification.{LogisticRegressionModel, SmoothedHingeLossLinearSVMModel}
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import com.linkedin.photon.ml.supervised.regression.{LinearRegressionModel, PoissonRegressionModel}
import com.linkedin.photon.ml.TaskType.TaskType

/**
 * Unit tests for TaskType.
 */
class TaskTypeTest {

  @DataProvider
  def matchCasesProvider(): Array[Array[Any]] =
    Array(
      Array(LinearRegressionModel(Coefficients(3)(1, 3, 5)(1.0, 3.0, 5.0)), TaskType.LINEAR_REGRESSION),
      Array(LogisticRegressionModel(Coefficients(3)(1, 3, 5)(1.0, 3.0, 5.0)), TaskType.LOGISTIC_REGRESSION),
      Array(PoissonRegressionModel(Coefficients(3)(1, 3, 5)(1.0, 3.0, 5.0)), TaskType.POISSON_REGRESSION),
      Array(SmoothedHingeLossLinearSVMModel(Coefficients(3)(1, 3, 5)(1.0, 3.0, 5.0)),
        TaskType.SMOOTHED_HINGE_LOSS_LINEAR_SVM)
    )

  @Test(dataProvider = "matchCasesProvider")
  def testTaskTypeMatch(model: GeneralizedLinearModel, modelType: TaskType): Unit =
    assertTrue(TaskType.matches(model, modelType))

  @DataProvider
  def mismatchCasesProvider(): Array[Array[Any]] =
    Array(
      Array(LinearRegressionModel(Coefficients(3)(1, 3, 5)(1.0, 3.0, 5.0)), TaskType.LOGISTIC_REGRESSION),
      Array(LogisticRegressionModel(Coefficients(3)(1, 3, 5)(1.0, 3.0, 5.0)), TaskType.LINEAR_REGRESSION),
      Array(PoissonRegressionModel(Coefficients(3)(1, 3, 5)(1.0, 3.0, 5.0)), TaskType.LOGISTIC_REGRESSION),
      Array(SmoothedHingeLossLinearSVMModel(Coefficients(3)(1, 3, 5)(1.0, 3.0, 5.0)), TaskType.LINEAR_REGRESSION)
    )

  @Test(dataProvider = "mismatchCasesProvider")
  def testTaskTypeMismatch(model: GeneralizedLinearModel, modelType: TaskType): Unit =
    assertFalse(TaskType.matches(model, modelType))
}
