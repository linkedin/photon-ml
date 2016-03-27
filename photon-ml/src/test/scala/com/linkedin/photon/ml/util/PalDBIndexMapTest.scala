package com.linkedin.photon.ml.util


import org.testng.annotations.Test

import org.testng.Assert._

/**
  * Created by yizhou on 3/23/16.
  */
class PalDBIndexMapTest {

  @Test
  def testMap(): Unit = {
    val map = new PalDBIndexMap().load("/tmp/index-output", 2)

    assertEquals(map.size(), 13)

    assertEquals(map.getIndex(new java.lang.String("2" + "\u0001")), 0)
    assertEquals(map.getIndex(new java.lang.String("8" + "\u0001")), 1)
    assertEquals(map.getIndex("5" + "\u0001"), 3)
    assertEquals(map.getIndex("9" + "\u0001"), 4)
  }
}
