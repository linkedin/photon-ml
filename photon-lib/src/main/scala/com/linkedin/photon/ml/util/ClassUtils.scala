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

/**
 * Object containing utilities for dealing with class types.
 */
object ClassUtils {

  private val ANON_CLASS_MARKER = "$anon$"

  /**
   * Method for detecting anonymous classes (the isAnonymousClass method is broken in Scala as late as 2.11).
   *
   * @param clazz A class
   * @return True if the given class is anonymous, false otherwise.
   */
  def isAnonClass(clazz: Class[_]): Boolean = clazz.getName.contains(ANON_CLASS_MARKER)

  /**
   * Get the true class type of an object, if it is an anonymous derived class.
   *
   * @tparam T Any type
   * @param obj An object
   * @return The object's class type
   */
  def getTrueClass[T](obj: T): Class[_] = {

    val clazz = obj.getClass

    if (isAnonClass(clazz)) {
      clazz.getSuperclass
    } else {
      clazz
    }
  }
}
