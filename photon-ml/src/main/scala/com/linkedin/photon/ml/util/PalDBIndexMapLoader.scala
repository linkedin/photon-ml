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


import java.net.URI

import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext


/**
 * A loader for IndexMaps stored in PalDB.
 */
class PalDBIndexMapLoader extends IndexMapLoader {

  private var _storeDir: String = _
  private var _numPartitions: Int = 0
  private var _namespace: String = _

  /**
   * Loads IndexMap data from PalDB files into Spark so that each executor node will have access to the correct
   * partition it needs (i.e. corresponding to the partition of the features it handles).
   *
   * @param sc the SparkContext
   * @param params the parameters object
   * @param namespace the namespace to use
   */
  override def prepare(sc: SparkContext, params: IndexMapParams, namespace: String = IndexMap.GLOBAL_NS): Unit = {

    val palDBParams = params match {
      case p: PalDBIndexMapParams => p
      case other =>
        throw new IllegalArgumentException(s"PalDBIndexMapLoader requires a params object of type " +
          s"PalDBIndexMapParams. ${other.getClass.getName}")
    }

    if (palDBParams.offHeapIndexMapDir.isDefined && palDBParams.offHeapIndexMapNumPartitions != 0) {

      _storeDir = palDBParams.offHeapIndexMapDir.get
      _numPartitions = palDBParams.offHeapIndexMapNumPartitions
      _namespace = namespace

      (0 until _numPartitions).foreach(i =>
        sc.addFile(getPath(sc, new Path(_storeDir, PalDBIndexMap.partitionFilename(i, namespace))))
      )
    } else {
      throw new IllegalArgumentException(s"offHeapIndexMapDir is empty or the offHeapIndexMapNumPartitions is zero." +
          s" Cannot init PalDBIndexMapLoader in this case.")
    }
  }

  override def indexMapForDriver(): IndexMap = new PalDBIndexMap().load(_storeDir, _numPartitions, _namespace)

  override def indexMapForRDD(): IndexMap = new PalDBIndexMap().load(_storeDir, _numPartitions, _namespace)

  /**
    * Converts a path object into a path string suitable for consumption by "sc.addFile". Implicitly converts a path
    * with no specified scheme to the current default scheme (sc.addFiles doesn't do this automatically).
    *
    * @param path the input path
    * @return the path string
    */
  protected[util] def getPath(sc: SparkContext, path: Path): String = {

    val uri = path.toUri

    Option(uri.getScheme) match {
      case Some(_) => uri.toString
      case _ =>
        // If the path specifies no scheme, use the current default
        val default = new Path(sc.hadoopConfiguration.get("fs.default.name")).toUri
        new URI(
          default.getScheme,
          default.getUserInfo,
          default.getHost,
          default.getPort,
          uri.getPath,
          uri.getQuery,
          uri.getFragment
        ).toString
    }
  }
}
