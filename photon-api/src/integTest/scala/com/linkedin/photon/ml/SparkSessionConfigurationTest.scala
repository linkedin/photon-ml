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
package com.linkedin.photon.ml

import org.apache.spark.SparkConf
import org.apache.spark.serializer.KryoSerializer
import org.testng.Assert._
import org.testng.annotations.Test

import com.linkedin.photon.ml.test.SparkTestUtils

/**
 * This class tests the SparkContextConfiguration object.
 */
class SparkSessionConfigurationTest extends SparkTestUtils {

  import SparkSessionConfiguration._

  // Synchronize across different potential SparkContext creators
  @Test
  def testAsYarnClient(): Unit = sparkTestSelfServeContext("testAsYarnClient") {
    val session1 = asYarnClient(new SparkConf().setMaster("local[1]"), "foo", useKryo = true)
    assertEquals(session1.sparkContext.getConf.get(CONF_SPARK_APP_NAME), "foo")
    assertEquals(session1.sparkContext.getConf.get(CONF_SPARK_SERIALIZER), classOf[KryoSerializer].getName)
    assertEquals(session1.sparkContext.getConf.get(CONF_SPARK_KRYO_CLASSES_TO_REGISTER),
      KRYO_CLASSES_TO_REGISTER.map(c => c.getName).mkString(",")
    )
    session1.stop()

    val session2 = asYarnClient(new SparkConf().setMaster("local[1]"), "bar", useKryo = false)
    assertEquals(session2.sparkContext.getConf.get(CONF_SPARK_APP_NAME), "bar")
    assertFalse(session2.sparkContext.getConf.contains(CONF_SPARK_SERIALIZER))
    assertFalse(session2.sparkContext.getConf.contains(CONF_SPARK_KRYO_CLASSES_TO_REGISTER))
    session2.stop()
  }
}
