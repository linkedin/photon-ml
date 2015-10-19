package com.linkedin.photon.ml

import com.linkedin.photon.ml.test.SparkTestUtils
import org.apache.spark.SparkConf
import org.apache.spark.serializer.KryoSerializer
import org.testng.Assert._
import org.testng.annotations.Test

/**
 * This class tests the SparkContextConfiguration object
 *
 * @author yizhou
 */
class SparkContextConfigurationIntegTest extends SparkTestUtils {

  import SparkContextConfiguration._

  // Synchronize across different potential SparkContext creators
  @Test
  def testAsYarnClient() = sparkTestSelfServeContext("testAsYarnClient") {
    val sc1 = asYarnClient(new SparkConf().setMaster("local[1]"), "foo", true)
    assertEquals(sc1.getConf.get(CONF_SPARK_APP_NAME), "foo")
    assertEquals(sc1.getConf.get(CONF_SPARK_SERIALIZER), classOf[KryoSerializer].getName())
    assertEquals(sc1.getConf.get(CONF_SPARK_KRYO_CLASSES_TO_REGISTER), KRYO_CLASSES_TO_REGISTER.map { case c => c.getName() }
        .mkString(","))
    sc1.stop()

    val sc2 = asYarnClient(new SparkConf().setMaster("local[1]"), "bar", false)
    assertEquals(sc2.getConf.get(CONF_SPARK_APP_NAME), "bar")
    assertFalse(sc2.getConf.contains(CONF_SPARK_SERIALIZER))
    assertFalse(sc2.getConf.contains(CONF_SPARK_KRYO_CLASSES_TO_REGISTER))
    sc2.stop()
  }
}
