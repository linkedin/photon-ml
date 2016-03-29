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


import com.linkedin.photon.ml.Params
import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext


/**
  * A PalDBIndexMap loader
  *
  */
class PalDBIndexMapLoader extends IndexMapLoader {
  private var _storeDir: String = null
  private var _numPartitions: Int = 0

  override def prepare(sc: SparkContext, params: Params): Unit = {
    if (!params.offHeapIndexMapDir.isEmpty && params.offHeapIndexMapNumPartitions != 0) {
      _storeDir = params.offHeapIndexMapDir.get
      _numPartitions = params.offHeapIndexMapNumPartitions
      (0 until _numPartitions).foreach(i =>
        sc.addFile(new Path(_storeDir, PalDBIndexMap.getPartitionFilename(i)).toUri().toString())
      )
    } else {
      throw new IllegalArgumentException(s"offHeapIndexMapDir is empty or the offHeapIndexMapNumPartitions is zero." +
          s" Cannot init PalDBIndexMapLoader in this case.")
    }
  }

  override def indexMapForDriver(): IndexMap = new PalDBIndexMap().load(_storeDir, _numPartitions)

  override def indexMapForRDD(): IndexMap = new PalDBIndexMap().load(_storeDir, _numPartitions)
}
