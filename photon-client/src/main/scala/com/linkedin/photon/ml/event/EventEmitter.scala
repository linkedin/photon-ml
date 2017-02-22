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
package com.linkedin.photon.ml.event

import scala.util.Try

// TODO: Log errors that occur during Event consumption

/**
 * Base trait for all [[Event]] producers.
 */
trait EventEmitter {
  private object Lock
  private var registeredListeners = List[EventListener]()

  /**
   * Register an [[EventListener]] to consume [[Event]] objects produced by this class.
   *
   * @param listener The listener
   */
  private[this] def registerListenerLocked(listener: EventListener) =
    registeredListeners = listener :: registeredListeners

  /**
   * Thread-safe and callable version of the above.
   *
   * @param listener The listener
   */
  protected def registerListener(listener: EventListener): Unit = Lock.synchronized(registerListenerLocked(listener))

  /**
   * Shutdown and remove all currently registered listeners.
   */
  private[this] def clearListenersLocked(): Unit = {
    registeredListeners.foreach { eventListener =>
      Try(eventListener.close())
    }
    registeredListeners = List()
  }

  /**
   * Thread-safe and callable version of the above.
   */
  protected def clearListeners(): Unit = Lock.synchronized(clearListenersLocked())

  /**
   * Produce an event for all registered listeners to consume.
   *
   * @param event The event
   */
  private[this] def sendEventLocked(event: Event): Unit = registeredListeners.foreach { eventListener =>
    Try(eventListener.handle(event))
  }

  /**
   * Thread-safe and callable version of the above.
   *
   * @param event The event
   */
  protected def sendEvent(event: Event): Unit = Lock.synchronized(sendEventLocked(event))
}
