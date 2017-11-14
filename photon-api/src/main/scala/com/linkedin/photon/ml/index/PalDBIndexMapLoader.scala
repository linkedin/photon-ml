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
package com.linkedin.photon.ml.index

import java.net.URI

import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.SparkContext

/**
 * A PalDBIndexMap loader.
 */
class PalDBIndexMapLoader(
  private val storeDir: String,
  private val numPartitions: Int,
  private val namespace: String = IndexMap.GLOBAL_NS) extends IndexMapLoader {

  /**
   * Load a feature index map, when in the driver.
   *
   * @return The loaded IndexMap for driver
   */
  override def indexMapForDriver(): IndexMap = new PalDBIndexMap().load(storeDir, numPartitions, namespace)

  /**
   * Load feature index map, when in an executor.
   *
   * @return The loaded IndexMap for RDDs
   */
  override def indexMapForRDD(): IndexMap = new PalDBIndexMap().load(storeDir, numPartitions, namespace)
}

object PalDBIndexMapLoader {
  /**
   * Converts a path object into a path string suitable for consumption by "sc.addFile". Implicitly converts a path
   * with no specified scheme to the current default scheme (sc.addFiles doesn't do this automatically).
   *
   * @param path The input path
   * @return The path string
   */
  protected[index] def getPath(sc: SparkContext, path: Path): String = {

    val uri = path.toUri

    Option(uri.getScheme) match {
      case Some(_) =>
        uri.toString

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

  /**
   * Factory methods for PalDBIndexMapLoaders.
   *
   * Throws exception if any parameter is not suitable to build a PalDBIndexMapLoader.
   *
   * @param sc The SparkContext
   * @param offHeapIndexMapDir The directory where the PalDB store will live (on HDFS)
   * @param numPartitions The number of partitions for the PalDB store
   * @param namespace A feature namespace, optional, defaults to IndexMap.GLOBAL_NS
   * @return An instance of a PalDBIndexMapLoader.
   */
  def apply(
      sc: SparkContext,
      offHeapIndexMapDir: Path,
      numPartitions: Int,
      namespace: String = IndexMap.GLOBAL_NS): PalDBIndexMapLoader = {

    val hadoopFS = FileSystem.get(sc.hadoopConfiguration)

    //
    // Pre-conditions
    //

    require(hadoopFS.exists(offHeapIndexMapDir), s"Off-heap index map dir does not exist: $offHeapIndexMapDir")
    require(
      hadoopFS.getContentSummary(offHeapIndexMapDir).getFileCount > 0,
      s"Off-heap index map dir is empty: $offHeapIndexMapDir")
    require(numPartitions > 0, s"Invalid # off-heap index map partitions: $numPartitions")

    (0 until numPartitions).foreach { i =>
      sc.addFile(
        PalDBIndexMapLoader.getPath(sc, new Path(offHeapIndexMapDir, PalDBIndexMap.partitionFilename(i, namespace))))
    }

    new PalDBIndexMapLoader(offHeapIndexMapDir.toString, numPartitions, namespace)
  }
}
