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
 * Configuration parameters for PalDBIndexMap.
 */
trait PalDBIndexMapParams {

  /**
   * The offheap storage directory if offheap map is needed. DefaultIndexMap will be used if not specified.
   */
  var offHeapIndexMapDir: Option[String] = None

  /**
   * The number of partitions for the offheap map storage. The partition number should be consistent with the number
   * when offheap storage is built. This parameter effects only the execution speed during feature index building and
   * has zero performance impact in training other than maintaining a convention.
   */
  var offHeapIndexMapNumPartitions: Int = 1
}
