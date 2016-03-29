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

import com.linkedin.paldb.api.{Configuration, PalDB, StoreReader}
import org.apache.spark.{SparkFiles, HashPartitioner}
import java.util.{Arrays => JArrays}
import java.io.{File => JFile}

/**
  * An off heap index map implementation using PalDB.
  *
  * The internal implementation assumed the following things:
  * 1. One DB storage is partitioned into multiple pieces we call partitions. It should be generated and controlled by
  * [[com.linkedin.photon.ml.FeatureIndexingJob]]. The partition strategy is via the hashcode of the feature names,
  * following the rules defined in [[org.apache.spark.HashPartitioner]].
  *
  * 2. Each time when a user is querying the index of a certain feature, the indexmap will first compute the hashcode,
  * and then compute the expected partition of the storageReader.
  *
  * 3. Because the way we are building each index partition (they are built in parallel, without sharing information
  * with each other). Each partition's internal index always starts from 0. Thus, we are keeping an offset array to
  * properly record how much offset we should provide for each index coming from a particular partition. In this way,
  * we could safely ensure that each index is unique.
  *
  * 4. Each time when a user is querying for the feature name of a given index, we'll do a linear search for each
  * storage until we find one or return null. TODO: this could be optimized further via a binary search of the offsets
  */
class PalDBIndexMap extends IndexMap {
  import PalDBIndexMap._

  @transient
  private[PalDBIndexMap] var _storeReaders: Array[StoreReader] = null
  private[PalDBIndexMap] var _offsets: Array[Int] = null

  private[PalDBIndexMap] var _partitionsNum: Int = 0
  private[PalDBIndexMap] var _size: Int = 0

  @transient
  private var _partitioner: HashPartitioner = null

  /**
    * Load a storage at a particular path
    *
    * @param storePath The directory where the storage is put
    * @param partitionsNum The number of partitions, the storage contains
    * @param isLocal default: false, if set false will use SparkFiles to access cached files; otherwise,
    *                it will directly read from local files
    * @return
    */
  def load(storePath: String, partitionsNum: Int, isLocal: Boolean = false): PalDBIndexMap = {
    _storeReaders = new Array[StoreReader](partitionsNum)
    _offsets = new Array[Int](partitionsNum)

    _partitionsNum = partitionsNum
    _partitioner = new HashPartitioner(_partitionsNum)

    for (i <- 0 until partitionsNum) {
      // Note: because we store both name -> idx and idx -> name in the same store
      _offsets(i) = _size / 2
      val filename = getPartitionFilename(i)

      val storeFile = if (isLocal) {
        new JFile(storePath, filename)
      } else {
        new JFile(SparkFiles.get(filename))
      }
      PALDB_READER_LOCK.synchronized {
        _storeReaders(i) = PalDB.createReader(storeFile, createDefaultPalDBConfig(_partitionsNum))
      }
      _size += _storeReaders(i).size().asInstanceOf[Number].intValue()
    }

    this
  }

  override def isEmpty(): Boolean = _storeReaders == null

  override def size(): Int = _size / 2

  override def getFeatureName(idx: Int): Option[String] = {
    var i = JArrays.binarySearch(_offsets, idx)
    // Note: check Arrays#binarySearch doc, >= 0 means we have a hit, otherwise it's (-insertion_pos-1)
    // insertion position is the 1st element that's greater than the key
    if (i < 0) {
      // The position before insertion position
      i = -(i + 1) - 1
    }

    if (i == -1) {
      // When the bounds are exceeded
      None
    } else {
      PALDB_READER_LOCK.synchronized {
        _storeReaders(i).get(idx - _offsets(i)).asInstanceOf[Any] match {
          case name: String => Some(name)
          case _ => None
        }
      }
    }
  }

  override def getIndex(name: String): Int = {
    val i = _partitioner.getPartition(name)
    // Note: very important to cast to java.lang.Object first; if directly casting to int,
    // null will be cast to 0 by Scala
    PALDB_READER_LOCK.synchronized {
      _storeReaders(i).get(name).asInstanceOf[Any] match {
        case idx: Int => idx + _offsets(i)
        case _ => IndexMap.NULL_KEY
      }
    }
  }

  //noinspection ScalaStyle
  override def +[B1 >: Int](kv: (String, B1)): Map[String, B1] = {
    throw new RuntimeException("This map is a read-only immutable map, add operation is unsupported.")
  }

  override def contains(key: String): Boolean = getIndex(key) >= 0

  override def get(key: String): Option[Int] = Option[Int](getIndex(key))

  override def iterator: Iterator[(String, Int)] = new PalDBIndexMapIterator(this)

  //noinspection ScalaStyle
  override def -(key: String): Map[String, Int] = {
    throw new RuntimeException("This map is a read-only immutable map, remove operation is unsupported.")
  }
}

object PalDBIndexMap {
  /* PalDB is not thread safe for parallel reads even for different storages, necessary to lock it.
   */
  private val PALDB_READER_LOCK = "READER_LOCK"
  // By default, we allow 200MB of LRU used by PalDBIndexMap in total
  private val DEFAULT_LRU_CACHE_BYTES = 209715200L

  /**
    * Returns the formatted filename for a partitioncular partition file of PalDB IndexMap.
    * This method should be used consistently as a protocol to handle naming conventions
    *
    * @param partitionId the partition Id
    * @return the formatted filename
    */
  def getPartitionFilename(partitionId: Int): String = s"paldb-partition-${partitionId}.dat"

  private def createDefaultPalDBConfig(partitionNum: Int): Configuration = {
    // TODO: make such config customizable in the future, so far, there isn't necessity for doing so.
    val config = new Configuration()
    config.set(Configuration.CACHE_ENABLED, "true")
    // Allow 200MB in-memory cache in total
    config.set(Configuration.CACHE_BYTES, String.valueOf(DEFAULT_LRU_CACHE_BYTES / partitionNum))
    config
  }

  class PalDBIndexMapIterator(private val map: PalDBIndexMap) extends Iterator[(String, Int)] {
    private var _i: Int = -1
    private var _currentItem: (String, Int) = null
    private var _currentStore: StoreReader = null
    private var _storeIterator: java.util.Iterator[java.util.Map.Entry[Any, Any]] = null

    override def hasNext: Boolean = {
      fetch()
      _currentItem != null
    }

    override def next(): (String, Int) = {
      fetch()
      if (_currentItem == null) {
        throw new RuntimeException("No more element exists.")
      }

      val res = _currentItem
      _currentItem = null
      res
    }

    private def fetch(): Unit = {
        while(_currentItem == null
            && ((_storeIterator != null && _storeIterator.hasNext) || _i < map._partitionsNum - 1)) {
          if (_storeIterator == null || !_storeIterator.hasNext) {
            _i += 1
            _currentStore = map._storeReaders(_i)
            _storeIterator = _currentStore.iterable().iterator()
          }

          while (_currentItem == null && _storeIterator.hasNext) {
            val entry = _storeIterator.next()
            if (entry.getKey.isInstanceOf[String]) {
              val idx = entry.getValue.asInstanceOf[Int]
              _currentItem = (entry.getKey().asInstanceOf[String], idx + map._offsets(_i))
            }
          }
        }
    }
  }
}
