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


import java.util.UUID

import com.linkedin.paldb.api.{PalDB, StoreWriter}
import org.apache.commons.io.FileUtils
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{Path, FileSystem}


/**
  * A IndexMapBuilder implemented via PalDB: [[https://github.com/linkedin/paldb]]
  *
  * @author yizhou
  */
class PalDBIndexMapBuilder extends IndexMapBuilder with Serializable {
  @transient
  private var _storeWriter: StoreWriter = null

  private var _i: Int = 0
  private var _tmpFile: java.io.File = null
  private var _dstFilePath: Path = null

  override def init(outputDir: String, partitionId: Int): IndexMapBuilder = {
    val filename = PalDBIndexMap.getPartitionFilename(partitionId)
    _tmpFile = new java.io.File(FileUtils.getTempDirectory, s"paldb-temp-${UUID.randomUUID().toString()}")
    _storeWriter = PalDB.createWriter(_tmpFile)
    _dstFilePath = new Path(outputDir, filename)

    this
  }

  override def put(name: String, idx: Int): IndexMapBuilder = {
    validateConfig()
    _storeWriter.put(name, idx)
    // Also store the reversed mapping
    _storeWriter.put(idx, name)

    if (idx >= _i) {
      _i = idx + 1
    }

    this
  }

  override def putIfAbsent(name: String): IndexMapBuilder = {
    put(name, _i)

    this
  }

  override def close(): Unit = {
    validateConfig()
    _storeWriter.close()

    val fs = FileSystem.get(new Configuration())
    fs.copyFromLocalFile(new Path(_tmpFile.toString()), _dstFilePath)
  }

  private def validateConfig(): Unit = {
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
  // Note: PalDB writer within the same JVM might stomp on each other and generate corrupted data, it's safer to
  // lock the write. This will only block writing operations within the same JVM
  val WRITER_LOCK: String = "DB_WRITER_LOCK"
}
