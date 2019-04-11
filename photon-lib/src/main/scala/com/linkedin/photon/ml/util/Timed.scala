/*
 * Copyright 2019 LinkedIn Corp. All rights reserved.
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

import java.util.concurrent.TimeUnit

import scala.concurrent.duration.{FiniteDuration, TimeUnit}

import org.slf4j.Logger

/**
 * An object to make it easy to time a block of code with minimal syntax clutter.
 * This method returns the value of the function timed, so that we can write, e.g.:
 *
 *  val featureIndexMapLoaders = Timed("prepare features") { prepareFeatureMaps() }
 *
 * Making it as easy as possible to add the cross-cutting concern of timing makes it
 * easy to collect basic profiling data (by named sections of code, where the dev
 * gets to define what is a meaningful block of code to "profile").
 */
object Timed {

  /**
   * This version of applies requires an explicit logger, msg and f to time.
   *
   * @param logger An (implicit) logger to which the timing will be output
   * @param msg    A message to output before the elapsed time
   * @param units  The units to use when reporting the elapsed time. The default is milliseconds.
   * @param f      The function/block of code... to time
   * @tparam T The type returned by f
   * @return The value returned by f, of type T
   */
  def apply[T](logger: Logger, msg: String, units: TimeUnit = TimeUnit.SECONDS)(f: => T): T =
    measureDuration(msg, units, f, logger)

  /**
   * This version allows to specify only the message to output with the timing.
   *
   * @param msg    A message to output before the elapsed time
   * @param f      The function/block of code... to time
   * @param logger An (implicit) logger to which the timing will be output
   * @tparam T The type returned by f
   * @return The value returned by f, of type T
   */
  def apply[T](msg: String)(f: => T)(implicit logger: Logger): T =
    measureDuration(msg, TimeUnit.SECONDS, f, logger)

  /**
   * A private method used by both "apply" to actually measure the duration and print out the result.
   *
   * @param msg    A message to output before the elapsed time
   * @param units  The units to use when reporting the elapsed time. The default is milliseconds.
   * @param f      The function/block of code... to time
   * @param logger An (implicit) logger to which the timing will be output
   * @tparam T The type returned by f
   * @return The value returned by f, of type T
   */
  protected[util] def measureDuration[T](msg: String, units: TimeUnit, f: => T, logger: Logger): T = {

    logger.info(s"$msg: begin execution")

    val t0 = System.nanoTime()
    val res = f
    val t1 = System.nanoTime()
    val duration = FiniteDuration(t1 - t0, TimeUnit.NANOSECONDS).toUnit(units)

    logger.info(s"$msg: executed in $duration $units")

    res
  }
}
