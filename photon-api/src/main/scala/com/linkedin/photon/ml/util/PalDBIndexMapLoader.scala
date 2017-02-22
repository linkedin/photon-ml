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

import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.SparkContext

/**
 * A PalDBIndexMap loader
 */
class PalDBIndexMapLoader(
  @transient sc: SparkContext,
  storeDir: String,
  numPartitions: Int,
  namespace: String = IndexMap.GLOBAL_NS) extends IndexMapLoader {

  private val _storeDir: String = storeDir
  private val _numPartitions: Int = numPartitions
  private val _namespace: String = namespace

  // Check the input arguments and add the PalDB index files to the Spark nodes.
  prepare()

  /**
   *
   */
  private[this] def prepare(): Unit = {
    val hadoopFS = FileSystem.get(sc.hadoopConfiguration)
    val offHeapIndexMapDirPath = new Path(_storeDir)

    // Check that the off-heap index map dir exists, is non-empty, and the loader is given a valid # of partitions.
    require(_numPartitions > 0, s"Invalid # off-heap index map partitions: ${_numPartitions}")
    require(hadoopFS.exists(offHeapIndexMapDirPath), s"Off-heap index map dir does not exist: ${_storeDir}")
    require(
      hadoopFS.getContentSummary(offHeapIndexMapDirPath).getFileCount > 0,
      s"Off-heap index map dir is empty: ${_storeDir}")

    (0 until _numPartitions).foreach { i =>
      sc.addFile(PalDBIndexMapLoader.getPath(sc, new Path(_storeDir, PalDBIndexMap.partitionFilename(i, namespace))))
    }
  }

  /**
   *
   * @return The loaded IndexMap for driver
   */
  override def indexMapForDriver(): IndexMap = new PalDBIndexMap().load(_storeDir, _numPartitions, _namespace)

  /**
   *
   * @return The loaded IndexMap for RDDs
   */
  override def indexMapForRDD(): IndexMap = new PalDBIndexMap().load(_storeDir, _numPartitions, _namespace)
}

object PalDBIndexMapLoader {
  /**
   * Converts a path object into a path string suitable for consumption by "sc.addFile". Implicitly converts a path
   * with no specified scheme to the current default scheme (sc.addFiles doesn't do this automatically).
   *
   * @param path The input path
   * @return The path string
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
          uri.getFragment).toString
    }
  }
}
