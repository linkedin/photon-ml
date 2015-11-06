package com.linkedin.photon.ml.data


import breeze.linalg.{DenseVector, SparseVector}
import com.linkedin.photon.ml.normalization.NormalizationContext
import com.linkedin.photon.ml.test.SparkTestUtils
import org.testng.Assert
import org.testng.annotations.{DataProvider, Test}


/**
 * Test BroadcastObjectProvider
 *
 * @author dpeng
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
      Array(NormalizationContext(factors = Some(DenseVector.ones[Double](5)), Some(SparseVector(5)((1, 3.0), (3, 0.2))), Some(2)))
    )
  }
}
