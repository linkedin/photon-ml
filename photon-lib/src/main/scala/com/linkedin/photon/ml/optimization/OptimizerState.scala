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
package com.linkedin.photon.ml.optimization

import breeze.linalg.Vector

/**
 * Similar to [[http://www.scalanlp.org/api/breeze/index.html#breeze.optimize.FirstOrderMinimizer\$State breeze.
 *   optimize.FirstOrderMinimizer.State]]
 *
 * This class tracks the information about the optimizer, including the coefficients, objective function value +
 * gradient, and the current iteration number.
 *
 * @param coefficients The current coefficients being optimized
 * @param loss The current objective function's value
 * @param gradient The current objective function's gradient
 * @param iter The current iteration number
 */
protected[optimization] case class OptimizerState(
    coefficients: Vector[Double],
    loss: Double,
    gradient: Vector[Double],
    iter: Int)
