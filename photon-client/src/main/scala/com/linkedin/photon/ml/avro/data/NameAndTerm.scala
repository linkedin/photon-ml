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
package com.linkedin.photon.ml.avro.data

/**
 * A compact way to represent the feature key as (name, term) pair.
 */
// TODO: Change the scope of this class and all functions in the companion object to [[com.linkedin.photon.ml.avro]]
// after Avro related classes/functions are decoupled from the rest of code
protected[ml] case class NameAndTerm(name: String, term: String) {

  /**
   *
   * @param that
   * @return
   */
  override def equals(that: Any): Boolean = that match {
    case other: NameAndTerm => name == other.name && term == other.term
    case _ => false
  }

  /**
   *
   * @return
   */
  override def hashCode: Int = (name + NameAndTerm.DELIMITER + term).hashCode

  /**
   *
   * @return
   */
  override def toString: String = s"name: $name, term: $term"
}

object NameAndTerm {
  private val DELIMITER = "\u0000"

  protected[ml] val INTERCEPT_NAME_AND_TERM = NameAndTerm("(INTERCEPT)", "")
}
