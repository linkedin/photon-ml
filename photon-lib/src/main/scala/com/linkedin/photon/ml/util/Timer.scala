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

import java.util.concurrent.atomic.AtomicBoolean

/**
 * Simple utility for making runtime performance measurements.
 *
 * Example usage:
 *
 *   val timer = new Timer
 *
 *   timer.start()
 *   // do stuff
 *   timer.stop()
 *   print(s"It took ${timer.durationSeconds} seconds!")
 *
 */
private[ml] class Timer {
  private val started: AtomicBoolean = new AtomicBoolean(false)
  private var startTime: Long = _
  private var stopTime: Long = _

  /**
   * Start the timer.
   *
   * @return The timer instance for chaining
   */
  def start(): Timer = {
    if (!started.compareAndSet(false, true)) {
      throw new IllegalStateException("Attempted to start timer when it's already started.")
    }

    startTime = now
    this
  }

  /**
   * Stop the timer.
   *
   * @return The timer instance for chaining
   */
  def stop(): Timer = {
    if (!started.compareAndSet(true, false)) {
      throw new IllegalStateException("Attempted to stop timer when it's already stopped.")
    }

    stopTime = now
    this
  }

  /**
   * Returns the current duration without affecting the state of the timer.
   *
   * @return The current duration
   */
  def mark: Long = {
    now - startTime
  }

  /**
   * Calculates current timer duration.
   *
   * @return The timer duration in nanoseconds
   */
  def duration: Long = {
    if (started.get) {
      throw new IllegalStateException("The timer must be stopped before reading its duration.")
    }

    stopTime - startTime
  }

  /**
   * Wraps the passed-in block with timing instrumentation, and returns a tuple of return value and duration.
   *
   * Example usage:
   *
   *   val (result, duration) = timer.measure {
   *     // do stuff
   *   }
   *   print(s"It took $duration nanos!")
   *
   * @param fn The block to execute
   * @return Tuple of (result, duration)
   */
  def measure[T](fn: => T): (T, Long) = {
    start()
    val result = fn
    stop()

    (result, duration)
  }

  /**
   * Wraps the passed-in block with timing instrumentation, and returns a tuple of return value and duration.
   *
   * Example usage:
   *
   *   val (result, duration) = timer.measureSeconds {
   *     // do stuff
   *   }
   *   print(s"It took $duration seconds!")
   *
   * @param fn The block to execute
   * @return Tuple of (result, duration in seconds)
   */
  def measureSeconds[T](fn: => T): (T, Double) = {
    val (result, duration) = measure(fn)
    (result, seconds(duration))
  }

  /**
   * Wraps the passed-in block with timing instrumentation, and returns a tuple of return value and duration.
   *
   * Example usage:
   *
   *   val (result, duration) = timer.measureMinutes {
   *     // do stuff
   *   }
   *   print(s"It took $duration minutes!")
   *
   * @param fn The block to execute
   * @return Tuple of (result, duration in minutes)
   */
  def measureMinutes[T](fn: => T): (T, Double) = {
    val (result, duration) = measure(fn)
    (result, minutes(duration))
  }

  /**
   * Calculates current timer duration.
   *
   * @return The timer duration in seconds
   */
  def durationSeconds: Double = seconds(duration)

  /**
   * Calculates current timer duration.
   *
   * @return The timer duration in minutes
   */
  def durationMinutes: Double = minutes(duration)

  /**
   * Returns the current time.
   *
   * @return Current time
   */
  private[util] def now: Long = System.nanoTime

  /**
   * Converts nanos to seconds.
   *
   * @param nanos Duration in nanoseconds
   * @return Duration in seconds
   */
  private def seconds(nanos: Long) = nanos * 1e-9

  /**
   * Converts nanos to minutes.
   *
   * @param nanos Duration in nanoseconds
   * @return Duration in minutes
   */
  private def minutes(nanos: Long) = seconds(nanos) / 60
}

object Timer {
  /**
   * Convenience function that creates and starts a timer.
   *
   * @return The started timer
   */
  def start(): Timer = (new Timer).start()

  /**
   * Wraps the passed-in block with timing instrumentation, and returns a tuple of return value and duration.
   *
   * Example usage:
   *
   *   val (result, duration) = Timer.measure {
   *     // do stuff
   *   }
   *   print(s"It took $duration nanos!")
   *
   *
   * @param fn The block to execute
   * @return Tuple of (result, duration)
   */
  def measure[T](fn: => T): (T, Long) = (new Timer).measure(fn)

  /**
   * Wraps the passed-in block with timing instrumentation, and returns a tuple of return value and duration.
   *
   * Example usage:
   *
   *   val (result, duration) = Timer.measureSeconds {
   *     // do stuff
   *   }
   *   print(s"It took $duration seconds!")
   *
   * @param fn The block to execute
   * @return Tuple of (result, duration in seconds)
   */
  def measureSeconds[T](fn: => T): (T, Double) = (new Timer).measureSeconds(fn)

  /**
   * Wraps the passed-in block with timing instrumentation, and returns a tuple of return value and duration.
   *
   * Example usage:
   *
   *   val (result, duration) = Timer.measureMinutes {
   *     // do stuff
   *   }
   *   print(s"It took $duration minutes!")
   *
   * @param fn The block to execute
   * @return Tuple of (result, duration in minutes)
   */
  def measureMinutes[T](fn: => T): (T, Double) = (new Timer).measureMinutes(fn)
}
