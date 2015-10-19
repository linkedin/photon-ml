package com.linkedin.photon.ml.optimization

import RegularizationType.RegularizationType
import org.testng.Assert
import org.testng.annotations.{DataProvider, Test}


/**
 * Test [[RegularizationContext]].
 *
 * @author dpeng
 */
class RegularizationContextTest {
  val epsilon = 1.0E-8
  @Test(dataProvider = "dataProvider")
  def testRegularizationContext(regularizationContext: RegularizationContext,
                                regularizationType: RegularizationType,
                                weight: Double,
                                l1Weight: Double,
                                l2Weight: Double): Unit = {
    Assert.assertEquals(regularizationContext.regularizationType, regularizationType)
    Assert.assertEquals(regularizationContext.getL1RegularizationWeight(weight), l1Weight, epsilon)
    Assert.assertEquals(regularizationContext.getL2RegularizationWeight(weight), l2Weight, epsilon)
  }

  @DataProvider(name = "dataProvider")
  def dataProvider(): Array[Array[Any]] = {
    val elastic1 = new RegularizationContext(RegularizationType.ELASTIC_NET, Some(0.5))
    val elastic2 = new RegularizationContext(RegularizationType.ELASTIC_NET, Some(0.1))
    Array(
      Array(L1RegularizationContext, RegularizationType.L1, 1.0d, 1.0d, 0.0d),
      Array(L2RegularizationContext, RegularizationType.L2, 1.0d, 0.0d, 1.0d),
      Array(elastic1, RegularizationType.ELASTIC_NET, 1.0d, 0.5d, 0.5d),
      Array(elastic2, RegularizationType.ELASTIC_NET, 2.0d, 0.2d, 1.8d)
    )
  }

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testL1WithAlpha(): Unit = {
    new RegularizationContext(RegularizationType.L1, Some(0.5))
  }

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testL2WithAlpha(): Unit = {
    new RegularizationContext(RegularizationType.L2, Some(0.5))
  }

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testElasticNetWithoutAlpha(): Unit = {
    new RegularizationContext(RegularizationType.ELASTIC_NET)
  }

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testTooLargeAlpha(): Unit = {
    new RegularizationContext(RegularizationType.ELASTIC_NET, Some(1.1d))
  }

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testNegativeAlpha(): Unit = {
    new RegularizationContext(RegularizationType.ELASTIC_NET, Some(-1d))
  }
}
