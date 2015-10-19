package com.linkedin.photon.ml

import org.testng.Assert._
import org.testng.annotations.Test

/**
 * Test DriverStage
 *
 * @author yizhou
 */
class DriverStageTest {

  @Test
  def testNames(): Unit = {
    assertEquals(DriverStage.INIT.name, "INIT")
    assertEquals(DriverStage.PREPROCESSED.name, "PREPROCESSED")
    assertEquals(DriverStage.TRAINED.name, "TRAINED")
    assertEquals(DriverStage.VALIDATED.name, "VALIDATED")
  }

  @Test
  def testOrderNum(): Unit = {
    assertEquals(DriverStage.INIT.order, 0)
    assertEquals(DriverStage.PREPROCESSED.order, 1)
    assertEquals(DriverStage.TRAINED.order, 2)
    assertEquals(DriverStage.VALIDATED.order, 3)
  }

  @Test
  def testOrder(): Unit = {
    assertTrue(DriverStage.INIT == DriverStage.INIT)
    assertTrue(DriverStage.PREPROCESSED == DriverStage.PREPROCESSED)
    assertTrue(DriverStage.TRAINED == DriverStage.TRAINED)
    assertTrue(DriverStage.VALIDATED == DriverStage.VALIDATED)

    assertTrue(DriverStage.INIT < DriverStage.VALIDATED)
    assertTrue(DriverStage.INIT <= DriverStage.VALIDATED)
    assertTrue(DriverStage.TRAINED > DriverStage.PREPROCESSED)
    assertTrue(DriverStage.TRAINED >= DriverStage.PREPROCESSED)
  }

  @Test
  def testSort(): Unit = {
    val stages1 = Array(DriverStage.TRAINED, DriverStage.VALIDATED, DriverStage.PREPROCESSED, DriverStage.INIT)
    assertEquals(stages1.sortWith(_ < _), Array(DriverStage.INIT, DriverStage.PREPROCESSED, DriverStage.TRAINED,
        DriverStage.VALIDATED))

    val stages2 = Array(DriverStage.INIT, DriverStage.PREPROCESSED, DriverStage.TRAINED)
    assertEquals(stages2.sortWith(_ < _), Array(DriverStage.INIT, DriverStage.PREPROCESSED, DriverStage.TRAINED))
  }
}
