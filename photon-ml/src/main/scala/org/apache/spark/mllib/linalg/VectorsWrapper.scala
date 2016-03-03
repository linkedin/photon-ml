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
package org.apache.spark.mllib.linalg
import breeze.linalg.{Vector => BV}

/**
 * This object is a wrapper to convert mllib vectors from/to breeze vectors. Due to the constraint from the mllib
 * package, the converters have very restricted access. This class bridges the gap so the converter becomes generally
 * available.
 *
 * @author dpeng
 */
object VectorsWrapper {
  def breezeToMllib(breezeVector: BV[Double]): Vector = {
    Vectors.fromBreeze(breezeVector)
  }

  def mllibToBreeze(mllibVector: Vector): BV[Double] = {
    mllibVector.toBreeze
  }
}
