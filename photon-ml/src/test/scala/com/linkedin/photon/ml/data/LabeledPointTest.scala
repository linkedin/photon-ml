package com.linkedin.photon.ml.data

import breeze.linalg.DenseVector
import com.linkedin.photon.ml.test.Assertions
import Assertions._
import com.linkedin.photon.ml.data
import org.testng.Assert._
import org.testng.annotations.Test

/**
 * Test the functions in [[data.LabeledPoint]]
 *
 * @author yali
 */
class LabeledPointTest
{
  import LabeledPoint._

  @Test
  def testApply(): Unit =
  {
    val delta = 1.0E-9
    val label = 1.0
    val features = DenseVector[Double](1.0, 10.0, 0.0, -100.0)
    val offset = 1.5
    val weight = 1.0
    val dataPoint = new LabeledPoint(label, features, offset, weight)
    val expected = LabeledPoint(label, features, offset, weight)
    assertEquals(dataPoint.label, expected.label, delta)
    assertIterableEqualsWithTolerance(dataPoint.features.toArray, expected.features.toArray, delta)
    assertEquals(dataPoint.offset, expected.offset, delta)
    assertEquals(dataPoint.weight, expected.weight, delta)
  }

  //test the unapply()
  @Test
  def testUnapply(): Unit = {
    val delta = 1.0E-9
    val label = 1.0
    val features = DenseVector[Double](12.21, 10.0, -0.03, 10.3)
    val offset = 1.5
    val weight = 3.2
    val dataPoint = LabeledPoint(label, features, offset, weight)
    val params = LabeledPoint.unapply(dataPoint)
    assertEquals(params.get._1, label, delta)
    assertIterableEqualsWithTolerance(params.get._2.toArray, features.toArray, delta)
    assertEquals(params.get._3, offset, delta)
    assertEquals(params.get._4, weight, delta)
  }

  //test the extractor by case class
  @Test
  def testExtractor(): Unit = {
    val delta = 1.0E-9
    val label = 1.0
    val features = DenseVector[Double](2.09, 113.0, -3.3, 150.30)
    val offset = 1.5
    val weight = 3.2
    val dataPoint = LabeledPoint(label, features, offset, weight)

    //test the extractor
    dataPoint match
    {
      case LabeledPoint(l, f, o, w) =>
      {
        assertEquals(l, label, delta)
        assertIterableEqualsWithTolerance(f.toArray, features.toArray, delta)
        assertEquals(o, offset, delta)
        assertEquals(w, weight, delta)
      }
      case _ => throw new RuntimeException(s"extractor behavior is unexpected : [$dataPoint]")
    }
  }
}
