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
 * This trait provides a uniform interface for loading feature index maps.
 *
 * To access an IndexMap within RDD operations, directly referring to an object inside Driver is inefficient.
 * The driver will try to serialize the entire object onto RDDs. This trait provides a uniform way of loading
 * feature index maps, regardless of their concrete implementation.
 */
trait IndexMapLoader extends Serializable {

  /**
   * To be called in the Driver. Whether or not this method returns a new instance or reuses an old one
   * depends on the implementor's decision.
   *
   * @return The loaded IndexMap for driver
   */
  def indexMapForDriver(): IndexMap

  /**
   * To be called in RDD operations. Whether or not this method returns a new instance or reuses an old one
   * should depend on the implementor's decision. This method should avoid serializing unnecessary large objects
   * to RDD executors.
   *
   * @return The loaded IndexMap for RDDs
   */
  def indexMapForRDD(): IndexMap
}
