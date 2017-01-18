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

import com.linkedin.photon.ml.supervised.classification.{LogisticRegressionModel, SmoothedHingeLossLinearSVMModel}
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import com.linkedin.photon.ml.supervised.regression.{LinearRegressionModel, PoissonRegressionModel}

/**
 * Type of models supported by Game
 */
object TaskType extends Enumeration {

  type TaskType = Value
  val LINEAR_REGRESSION, POISSON_REGRESSION, LOGISTIC_REGRESSION, SMOOTHED_HINGE_LOSS_LINEAR_SVM, NONE = Value

  /**
   * A method that decides which value of this TaskType enumeration the model corresponds to.
   *
   * @param model The model to look at
   * @return A TaskType value for the model
   */
  private def switch(model: GeneralizedLinearModel): TaskType =
    model match {
      case _: LinearRegressionModel => LINEAR_REGRESSION
      case _: LogisticRegressionModel => LOGISTIC_REGRESSION
      case _: PoissonRegressionModel => POISSON_REGRESSION
      case _: SmoothedHingeLossLinearSVMModel => SMOOTHED_HINGE_LOSS_LINEAR_SVM
      case _ => NONE
    }

  /**
   * Checks if a subclass of GeneralizedLinearModel corresponds to the expected TaskType.
   *
   * @param model The model to check
   * @param taskType The expected task type
   * @return True if the model corresponds to the expected task type, false otherwise
   */
  def matches(model: GeneralizedLinearModel, taskType: TaskType): Boolean = switch(model) == taskType

  /**
   * Return a human readable name for a TaskType value.
   *
   * @param model The model for which we want a model type
   * @return The model type as a String
   */
  def name(model: GeneralizedLinearModel): String = switch(model).toString
}
