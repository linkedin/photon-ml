/*
 * Copyright 2017 LinkedIn Corp. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License"); you may
 * not use this file except in compliance with the License. You may obtain a
 * copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */
package com.linkedin.photon.ml.util

import java.util.NoSuchElementException

import org.testng.Assert._
import org.testng.annotations.Test

import com.linkedin.photon.ml.util.Implicits._

/**
 * Unit tests for Implicits.
 */
class ImplicitsTest {

  /**
   * Check that we can chain a "tap" and continue on.
   */
  @Test
  def testTapMap(): Unit = {

    val (pwi, pwo) = (new StringBuilder, new StringBuilder)

    Map('a' -> 1, 'b' -> 2, 'c' -> 3)
      .tap(el => pwi.append(el._2 + 1))
      .foreach(el => pwo.append(s"${el._1}:${el._2} "))

    assertEquals(pwi.toString, "234")
    assertEquals(pwo.toString, "a:1 b:2 c:3 ")
  }

  /**
   * Check that we can tap keys only.
   */
  @Test
  def testTapMapKeys(): Unit = {

    val (pwi, pwo) = (new StringBuilder, new StringBuilder)

    Map('a' -> 1, 'b' -> 2, 'c' -> 3)
      .tapKeys(pwi.append(_))
      .foreach(el => pwo.append(s"${el._1}:${el._2} "))

    assertEquals(pwi.toString, "abc")
    assertEquals(pwo.toString, "a:1 b:2 c:3 ")
  }

  /**
   * Check that we can tap values only.
   */
  @Test
  def testTapMapValues(): Unit = {

    val (pwi, pwo) = (new StringBuilder, new StringBuilder)

    Map('a' -> 1, 'b' -> 2, 'c' -> 3)
      .tapValues(pwi.append(_))
      .foreach(el => pwo.append(s"${el._1}:${el._2} "))

    assertEquals(pwi.toString, "123")
    assertEquals(pwo.toString, "a:1 b:2 c:3 ")
  }

  /**
   * Check that we can tap a List.
   */
  @Test
  def testTapList(): Unit = {

    val (pwi, pwo) = (new StringBuilder, new StringBuilder)

    List(1, 2, 3)
      .tap(el => pwi.append(el + 1))
      .foreach(el => pwo.append(s"$el:"))

    assertEquals(pwi.toString, "234")
    assertEquals(pwo.toString, "1:2:3:")
  }

  /**
   * Check that we can tap an Array.
   */
  @Test
  def testTapArray(): Unit = {

    val (pwi, pwo) = (new StringBuilder, new StringBuilder)

    Array(1, 2, 3)
      .tap(el => pwi.append(el + 1))
      .foreach(el => pwo.append(s"$el:"))

    assertEquals(pwi.toString, "234")
    assertEquals(pwo.toString, "1:2:3:")
  }

  /**
   * Check that we can tap a Set.
   */
  @Test
  def testTapSet(): Unit = {

    val (pwi, pwo) = (new StringBuilder, new StringBuilder)

    Set(1, 2, 3)
      .tap(el => pwi.append(el + 1))
      .foreach(el => pwo.append(s"$el:"))

    assertEquals(pwi.toString, "234")
    assertEquals(pwo.toString, "1:2:3:")
  }

  /**
   * Check that we can tap an Iterable.
   */
  @Test
  def testTapIterable(): Unit = {

    val (pwi, pwo) = (new StringBuilder, new StringBuilder)

    Map('a' -> 1, 'b' -> 2, 'c' -> 3)
      .values.map(v => v + 1) // .values, like .keys, returns an Iterable[T]
      .tap(el => pwi.append(el - 1))
      .foreach(el => pwo.append(s"$el:"))

    assertEquals(pwi.toString, "123")
    assertEquals(pwo.toString, "2:3:4:")
  }

  /**
   * Check that we can tap an Option, both defined and empty.
   */
  @Test
  def tapOption(): Unit = {

    val pwo = new StringBuilder

    assertTrue(Option(List(1,2,3)).tap(l => l.foreach(pwo.append)).isDefined)
    assertEquals(pwo.toString, "123")

    var str = ""
    assertTrue(Option[String](null).tap(_ => str += "nope").isEmpty)
    assertEquals(str, "")
  }

  /**
   * Check that we can extract from a Map 'through' an Option.
   */
  @Test
  def testExtractOrElse(): Unit = {

    assertEquals(Some(Map('a' -> 1, 'b' -> 2)).extractOrElse('a')(0), 1)
    assertEquals(Some(Map('a' -> 1, 'b' -> 2)).extractOrElse('b')(0), 2)
    assertEquals(Option.empty[Map[Char, Int]].extractOrElse('a')(0), 0)

    try {
      assertEquals(Some(Map('a' -> 1, 'b' -> 2)).extractOrElse('c')(0), 2)
    } catch {
      case _: NoSuchElementException => // expected, throw by Map as usual when key not found
      case e: Throwable => throw e // bad
    }

  }
}
