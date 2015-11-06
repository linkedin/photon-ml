package com.linkedin.photon.ml.data


import breeze.linalg.{DenseVector, SparseVector}
import com.linkedin.photon.ml.normalization.NormalizationContext
import org.testng.Assert
import org.testng.annotations.{DataProvider, Test}


/**
 * Test SimpleObjectProvider
 *
 * @author dpeng
 */
class SimpleObjectProviderTest {
  @Test(dataProvider = "dataProvider")
  def testSimpleObjectProvider(obj: Serializable): Unit = {
    val provider = new SimpleObjectProvider[Serializable](obj)
    Assert.assertEquals(provider.get, obj)
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
