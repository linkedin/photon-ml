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

import java.util.UUID

import com.linkedin.paldb.api.{PalDB, StoreWriter}
import org.apache.commons.io.FileUtils
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}

/**
 * A IndexMapBuilder implemented via PalDB: [[https://github.com/linkedin/paldb]].
 */
class PalDBIndexMapBuilder extends IndexMapBuilder with Serializable {

  // The nulls here are intentional.
  @transient
  private var _storeWriter: StoreWriter = null
  private var _tmpFile: java.io.File = null
  private var _dstFilePath: Path = null

  /**
   * Init a builder that will store feature name and indexes into a PalDB store.
   *
   * @param outputDir The HDFS directory to store the built index map file
   * @param partitionId The partition id of current builder
   * @param namespace A feature namespace
   * @return The current builder
   */
  override def init(outputDir: String, partitionId: Int, namespace: String): IndexMapBuilder = {
    val filename = PalDBIndexMap.partitionFilename(partitionId, namespace)
    _tmpFile = new java.io.File(FileUtils.getTempDirectory, s"paldb-temp-${UUID.randomUUID().toString}")
    _storeWriter = PalDB.createWriter(_tmpFile)
    _dstFilePath = new Path(outputDir, filename)
    checkInvariants()
    this
  }

  /**
   * Store feature name and its index. The feature name and index are stored only in a writer.
   *
   * @param name The feature name
   * @param idx The feature index
   * @return The current builder
   */
  override def put(name: String, idx: Int): IndexMapBuilder = {
    _storeWriter.put(name, idx)
    // Also store the reversed mapping
    _storeWriter.put(idx, name)
    this
  }

  /**
   * Upon close(), the writer's content actually goes to the PalDB store on disk.
   */
  override def close(): Unit = {
    _storeWriter.close()
    val fs = FileSystem.get(new Configuration())
    fs.copyFromLocalFile(new Path(_tmpFile.toString), _dstFilePath)
  }

  /**
   * Check the invariants that must hold for this object to be usable.
   * Throws exceptions if this feature index map builder is not set up properly.
   */
  private def checkInvariants(): Unit = {
    if (_storeWriter == null) {
      throw new RuntimeException("Cannot proceed, storeWriter is null.")
    }

    if (_tmpFile == null || !_tmpFile.exists()) {
      throw new RuntimeException(s"Cannot proceed, tmpFile is null or does not exist.")
    }

    if (_dstFilePath == null) {
      throw new RuntimeException("Cannot proceed, the to-save output path is null.")
    }
  }
}

object PalDBIndexMapBuilder {
  // NOTE PalDB writer within the same JVM might stomp on each other and generate corrupted data, it's safer to
  // lock the write. This will only block writing operations within the same JVM
  val WRITER_LOCK: String = "DB_WRITER_LOCK"
}
