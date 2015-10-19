package com.linkedin.photon.ml.optimization

import breeze.linalg.{DenseVector, SparseVector, Vector}
import org.testng.Assert
import org.testng.annotations.{DataProvider, Test}


/**
 * @author nkatariy
 */
class OptimizationUtilsTest {
  @DataProvider
  def generateCoefficientsAndConstraintMap(): Array[Array[Object]] = {
    val dVec = DenseVector(0.0, 1.0, -1.0, 0.0, 5.0)
    val sVec = SparseVector(0.0, 1.0, -1.0, 0.0, 5.0)

    Array(
      Array(dVec, None, dVec),
      Array(sVec, None, sVec),
      Array(dVec, Some(Map[Int, (Double, Double)]()), dVec),
      Array(sVec, Some(Map[Int, (Double, Double)]()), sVec),
      Array(dVec, Some(Map[Int, (Double, Double)](1->(-0.5, 0.5), 4->(6.7, Double.PositiveInfinity))),
        DenseVector(0.0, 0.5, -1.0, 0.0, 6.7)),
      Array(sVec, Some(Map[Int, (Double, Double)](1->(-0.5, 0.5), 4->(6.7, Double.PositiveInfinity))),
        SparseVector(0.0, 0.5, -1.0, 0.0, 6.7)),
      Array(dVec,
        Some(Map[Int, (Double, Double)](0->(-1.0, 0.0), 1->(-0.5, 0.5), 2->(0.0, 1.0), 3->(Double.NegativeInfinity, 0.0),
          4->(6.7, Double.PositiveInfinity))),
        DenseVector(0.0, 0.5, 0.0, 0.0, 6.7)),
      Array(sVec,
        Some(Map[Int, (Double, Double)](0->(-1.0, 0.0), 1->(-0.5, 0.5), 2->(0.0, 1.0), 3->(Double.NegativeInfinity, 0.0),
          4->(6.7, Double.PositiveInfinity))),
        SparseVector(0.0, 0.5, 0.0, 0.0, 6.7))
    )
  }

  @Test(dataProvider = "generateCoefficientsAndConstraintMap")
  def testProjectCoefficientsToHypercube(coefficients: Vector[Double], constraints: Option[Map[Int, (Double, Double)]],
                                            expectedVectorOutput: Vector[Double]) = {
    Assert.assertEquals(OptimizationUtils.projectCoefficientsToHypercube(coefficients, constraints), expectedVectorOutput)
  }
}