package com.linkedin.photon.ml.test

import org.testng.Assert.assertEquals


/**
 * This object provides utility for complex assertions.
 * @author dpeng
 */
object Assertions {

  /**
   * This method compares lists of doubles with a tolerance.
   * @param actual The actual iterable
   * @param expected The expected iterable
   * @param delta The tolerance
   */
  def assertIterableEqualsWithTolerance(actual: Iterable[Double], expected: Iterable[Double], delta: Double): Unit = {
    actual.zip(expected).foreach {
      case (act, exp) => assertEquals(act, exp, delta)
    }
  }
}
