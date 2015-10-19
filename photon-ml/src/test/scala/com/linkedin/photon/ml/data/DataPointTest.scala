package com.linkedin.photon.ml.data

import breeze.linalg.DenseVector
import com.linkedin.photon.ml.test.Assertions
import Assertions._
import com.linkedin.photon.ml.data
import org.testng.Assert._
import org.testng.annotations.Test


/**
 * Test the functions in [[data.DataPoint]]
 *
 * @author yali
 */
class DataPointTest {
  import DataPoint._

  //test the class and object
  @Test
  def testApply(): Unit = {
    val delta = 1.0E-9
    val features = DenseVector[Double](1.0, 10.0, 0.0, -100.0)
    val weight = 1.0
    val dataPoint = new DataPoint(features, weight)
    val expected = DataPoint(features, weight)
    assertIterableEqualsWithTolerance(dataPoint.features.toArray, expected.features.toArray, delta)
    assertEquals(dataPoint.weight, expected.weight, delta)
  }

  //test unapply()
  @Test
  def testUnapply(): Unit = {
    val delta = 1.0E-9
    val features = DenseVector[Double](1.5, 13.0, -3.3, 1350.02)
    val weight = 3.2
    val dataPoint = DataPoint(features, weight)
    val featuresAndWeight = DataPoint.unapply(dataPoint)
    assertIterableEqualsWithTolerance(dataPoint.features.toArray, featuresAndWeight.get._1.toArray, delta)
    assertEquals(dataPoint.weight, featuresAndWeight.get._2, delta)
  }

  //test the extractor by case class
  @Test
  def testExtractor(): Unit = {
    val delta = 1.0E-9
    val features = DenseVector[Double](2.09, 113.0, -3.3, 150.30)
    val weight = 6.4
    val dataPoint = DataPoint(features, weight)

    //test the extractor
    dataPoint match
    {
      case DataPoint(f, w) =>
      {
        assertEquals(dataPoint.weight, w, delta)
        assertIterableEqualsWithTolerance(dataPoint.features.toArray, f.toArray, delta)
      }
      case _ => throw new RuntimeException(s"extractor behavior is unexpected : [$dataPoint]")
    }
  }
}