package com.linkedin.photon.ml.function


import org.testng.Assert
import org.testng.annotations.{DataProvider, Test}


/**
 * This tests the validity of first and second derivatives of single loss functions
 * using finite difference.
 *
 * @author dpeng
 */
class PointwiseLossFunctionTest {
  val delta = 1.0E-3
  val epsilon = 1.0E-6

  @DataProvider
  def dataProvider(): Array[Array[Any]] = {
    val dataSource = Array(-1.0, 0.0, 1.0)
    val funcs = Array(PointwiseLogisticLossFunction, PointwisePoissonLossFunction, PointwiseSquareLossFunction)
    for {x <- dataSource; y <- dataSource; func <- funcs } yield Array(func, x, y)
  }

  @Test(dataProvider = "dataProvider")
  def testGradient(func: PointwiseLossFunction, margin: Double, label: Double): Unit = {
    val (_, grad) = func.loss(margin, label)
    val (valuep, gradp) = func.loss(margin + delta, label)
    val (valuem, gradm) = func.loss(margin - delta, label)
    val testG = (valuep - valuem) / 2 / delta
    Assert.assertEquals(testG, grad, epsilon)

    val hess = func.d2lossdz2(margin, label)
    val testH = (gradp - gradm) / 2 / delta
    Assert.assertEquals(testH, hess, epsilon)
  }
}
