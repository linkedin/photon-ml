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
class PalDBIndexMapBuilder extends IndexMapBuilder with java.io.Serializable {
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
    _storeWriter.close()

    val fs = FileSystem.get(new Configuration())
    fs.copyFromLocalFile(new Path(_tmpFile.toString()), _dstFilePath)
  }
}

object PalDBIndexMapBuilder {
  // Note: PalDB writer within the same JVM might stomp on each other and generate corrupted data, it's safer to
  // lock the write. This will only block writing operations within the same JVM
  val WRITER_LOCK: String = "DB_WRITER_LOCK"
}
