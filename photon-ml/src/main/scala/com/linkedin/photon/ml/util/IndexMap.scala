/*
 * Copyright 2016 LinkedIn Corp. All rights reserved.
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


import scala.collection.immutable.Map

/**
  * The trait defines the methods supposed should be supported by an index map
  *
  * @author yizhou
  */
trait IndexMap extends Map[String, Int] with Serializable {

  /**
    * Given an index, reversely track down the corresponding feature name
    *
    * @param idx the feature index
    * @return the feature name, return null if not found
    */
  def getFeatureName(idx: Int): String

  /**
    * Given a feature string, return the index
    *
    * @param name the feature name
    * @return the feature index, return IndexMap.NULL_KEY if not found
    */
  def getIndex(name: String): Int
}

object IndexMap {
  // The key to indicate a feature is not existing in the map
  val NULL_KEY:Int = -1
}
