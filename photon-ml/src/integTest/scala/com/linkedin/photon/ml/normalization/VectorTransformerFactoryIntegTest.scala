package com.linkedin.photon.ml.normalization

import java.lang.{Object => JObject}

import breeze.linalg.{DenseVector, SparseVector}
import com.linkedin.photon.ml.test.{Assertions, SparkTestUtils}
import Assertions.assertIterableEqualsWithTolerance
import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.test.SparkTestUtils
import org.testng.annotations.{DataProvider, Test}

/**

 * @author dpeng
 */
class VectorTransformerFactoryIntegTest extends SparkTestUtils {
  /*
   * features:
   *  1  1  1  1  1  0
   *  2  0 -1  0  1  0
   * .2  0 .5  0  1  0
   *  0 10  0  5  1  0
   *
   *  The fifth and sixth columns should be left untouched in all scalers.
   */
  private val _features = Array(
    DenseVector[Double](1.0, 1.0, 1.0, 1.0, 1.0, 0.0),
    DenseVector[Double](2.0, 0.0, -1.0, 0.0, 1.0, 0.0),
    new SparseVector[Double](Array(0, 2, 4), Array(0.2, 0.5, 1.0), 6),
    new SparseVector[Double](Array(1, 3, 4), Array(10.0, 5.0, 1.0), 6)
  )

  private val _delta = 1.0E-5
  private val _stdFactors = Array(1.09985, 0.20592, 1.17108, 0.42008, 1.0, 1.0)
  private val _maxMagnitudeFactors = Array(0.5, 0.1, 1.0, 0.2, 1.0, 1.0)
  private val _identityFactors = Array(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)

  @Test(dataProvider = "generateTestData")
  def testFactor(normalizationType: NormalizationType, expectedFactors: Array[Double]): Unit = sparkTest("test") {
    val rdd = sc.parallelize(_features.map(x => new LabeledPoint(0, x)))
    val scaler = VectorTransformerFactory(normalizationType, rdd)
    val ones = DenseVector.ones[Double](6)
    val factors = scaler.transform(ones)
    assertIterableEqualsWithTolerance(factors.toArray, expectedFactors, _delta)
  }

  @DataProvider
  def generateTestData(): Array[Array[JObject]] = {
    Array(
      Array(NormalizationType.USE_STANDARD_DEVIATION, _stdFactors),
      Array(NormalizationType.USE_MAX_MAGNITUDE, _maxMagnitudeFactors),
      Array(NormalizationType.NO_SCALING, _identityFactors)
    )
  }
}
