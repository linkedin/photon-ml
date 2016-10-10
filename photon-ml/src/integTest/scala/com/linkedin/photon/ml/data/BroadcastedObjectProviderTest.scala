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
package com.linkedin.photon.ml.data


import breeze.linalg.{DenseVector, SparseVector}
import com.linkedin.photon.ml.normalization.NormalizationContext
import com.linkedin.photon.ml.test.SparkTestUtils
import org.testng.Assert
import org.testng.annotations.{DataProvider, Test}


/**
 * Test BroadcastObjectProvider
 */
class BroadcastedObjectProviderTest extends SparkTestUtils {
  @Test(dataProvider = "dataProvider")
  def testSimpleObjectProvider(obj: Serializable): Unit = sparkTest("testSimpleObjectProvider") {
    val broadcast = sc.broadcast(obj)
    val provider = new BroadcastedObjectProvider[Serializable](broadcast)
    Assert.assertEquals(provider.get, obj)
    broadcast.unpersist()
  }

  @DataProvider
  def dataProvider(): Array[Array[Any]] = {
    Array(
      Array(SparseVector(5)((1, 3.0), (3, 0.2))),
      Array(DenseVector.ones[Double](5)),
      Array(NormalizationContext(
        factors = Some(DenseVector.ones[Double](5)),
        Some(SparseVector(5)((1, 3.0), (3, 0.2))),
        Some(2)
      ))
    )
  }
}
