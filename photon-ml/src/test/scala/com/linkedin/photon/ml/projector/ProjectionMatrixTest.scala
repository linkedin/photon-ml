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
package com.linkedin.photon.ml.projector

import breeze.linalg.{CSCMatrix, DenseMatrix, DenseVector}
import org.testng.Assert
import org.testng.annotations.Test


class ProjectionMatrixTest {
   // matrix is [[1.0, 0.0, 3.0], [0.0, 2.0, 4.0]]
   val projectionMatrix = new ProjectionMatrix(new DenseMatrix[Double](2, 3, Array(1.0, 0.0, 0.0, 2.0, 3.0, 4.0)))

   @Test(expectedExceptions = Array(classOf[UnsupportedOperationException]))
   def testConstructorWithUnsupportedMatrixType() = {
     val _ = new ProjectionMatrix(CSCMatrix.zeros[Double](2, 3))
   }

   @Test
   def testProjectFeatures() = {
     // projection should be [10.0, 16.0]
     val v = new DenseVector[Double](Array(1.0, 2.0, 3.0))
     Assert.assertEquals(projectionMatrix.projectFeatures(v).iterator.toSet, Set[(Int, Double)]((0, 10.0), (1, 16.0)))
   }

   @Test
   def testProjectCoefficients() = {
     // projection should be [-2, 8, 10]
     val coefficients = new DenseVector[Double](Array(-2.0, 4.0))
     Assert.assertEquals(projectionMatrix.projectCoefficients(coefficients).iterator.toSet,
       Set[(Int, Double)]((0, -2.0), (1, 8.0), (2, 10.0)))
   }

   // TODO Test gaussian random projection matrix. Current impl looks strange w.r.t the conventional way of creating
   // random projection matrices and there are no references for how it has actually been implemented.
 }