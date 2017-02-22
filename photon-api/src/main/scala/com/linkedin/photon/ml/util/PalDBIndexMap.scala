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

import java.io.{File => JFile}
import java.util.{Arrays => JArrays, Map => JMap}

import collection.JavaConverters._

import com.linkedin.paldb.api.{Configuration, PalDB, StoreReader}
import org.apache.spark.{HashPartitioner, SparkFiles}

/**
 * An off heap index map implementation using [[PalDB]].
 *
 * The internal implementation assumed the following things:
 * 1. One DB storage is partitioned into multiple pieces we call partitions. It should be generated and controlled by
 * [[com.linkedin.photon.ml.FeatureIndexingJob]]. The partition strategy is via the hashcode of the feature names,
 * following the rules defined in [[org.apache.spark.HashPartitioner]].
 *
 * 2. Each time when a user is querying the index of a certain feature, the index map will first compute the hashcode,
 * and then compute the expected partition of the storageReader.
 *
 * 3. Because the way we are building each index partition (they are built in parallel, without sharing information
 * with each other). Each partition's internal index always starts from 0. Thus, we are keeping an offset array to
 * properly record how much offset we should provide for each index coming from a particular partition. In this way,
 * we could safely ensure that each index is unique.
 *
 * 4. Each time when a user is querying for the feature name of a given index, we'll do a binary search for the proper
 * storage according to offset ranges and then return null or the proper feature name.
 */
class PalDBIndexMap extends IndexMap {
  import PalDBIndexMap._

  @transient
  private[PalDBIndexMap] var _storeReaders: Array[StoreReader] = _
  // Each store's internal indices start from 0, this offsets the external returned idx should be
  //  ([internal_idx] + offset) so that it is always globally unique
  private[PalDBIndexMap] var _offsets: Array[Int] = _

  private[PalDBIndexMap] var _partitionsNum: Int = 0
  // Internal use only, total number of elements, including both id -> name and name -> id mappings
  private[PalDBIndexMap] var _size: Int = 0

  @transient
  private var _partitioner: HashPartitioner = _

  /**
   * Load a storage at a particular path
   *
   * @param storePath The directory where the storage is put
   * @param partitionsNum The number of partitions, the storage contains
   * @param namespace
   * @param isLocal default: false, if set false will use SparkFiles to access cached files; otherwise,
   *                it will directly read from local files
   * @return A PalDBIndexMap instance
   */
  def load(
      storePath: String,
      partitionsNum: Int,
      namespace: String = IndexMap.GLOBAL_NS,
      isLocal: Boolean = false): PalDBIndexMap = {

    _storeReaders = new Array[StoreReader](partitionsNum)
    _offsets = new Array[Int](partitionsNum)

    _partitionsNum = partitionsNum
    _partitioner = new HashPartitioner(_partitionsNum)

    for (i <- 0 until partitionsNum) {
      // Note: because we store both name -> idx and idx -> name in the same store
      _offsets(i) = _size / 2
      val filename = partitionFilename(i, namespace)

      val storeFile = if (isLocal) {
        new JFile(storePath, filename)
      } else {
        new JFile(SparkFiles.get(filename))
      }
      PALDB_READER_LOCK.synchronized {
        // TODO: make such config customizable in the future, so far, there isn't necessity for doing so.
        _storeReaders(i) = PalDB.createReader(storeFile, new Configuration())
      }
      _size += _storeReaders(i).size().asInstanceOf[Number].intValue()
    }

    this
  }

  /**
   *
   * @return
   */
  override def isEmpty(): Boolean = _storeReaders == null

  /**
   *
   * @return
   */
  override def size(): Int = _size / 2

  /**
   *
   * @param idx The feature index
   * @return The feature name if found, NONE otherwise
   */
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

  /**
   *
   * @param name The feature name
   * @return The feature index if found, IndexMap.NULL_KEY otherwise
   */
  override def getIndex(name: String): Int = {
    val i = getPartitionId(name)
    // Note: very important to cast to java.lang.Object first; if directly casting to int,
    // null will be cast to 0 by Scala
    PALDB_READER_LOCK.synchronized {
      _storeReaders(i).get(name).asInstanceOf[Any] match {
        case idx: Int => idx + _offsets(i)
        case _ => IndexMap.NULL_KEY
      }
    }
  }

  /**
   *
   * @param name
   * @return
   */
  private[PalDBIndexMap] def getPartitionId(name: String): Int = {
    _partitioner.getPartition(name)
  }

  /**
   *
   * @param kv
   * @tparam B1
   * @return
   */
  override def +[B1 >: Int](kv: (String, B1)): Map[String, B1] =
    throw new RuntimeException("This map is a read-only immutable map, add operation is unsupported.")

  /**
   *
   * @param key
   * @return
   */
  override def -(key: String): Map[String, Int] =
    throw new RuntimeException("This map is a read-only immutable map,remove operation is unsupported.")

  /**
   *
   * @param key
   * @return
   */
  override def contains(key: String): Boolean = getIndex(key) >= 0

  /**
   *
   * @param key
   * @return
   */
  override def get(key: String): Option[Int] = Option[Int](getIndex(key))

  /**
   *
   * @note This method is thread unsafe, external synchronization should be handled properly
   *       in concurrent settings.
   *
   * @return An iterator walking through all stored feature mappings as (name, index) tuples
   */
  override def iterator: Iterator[(String, Int)] = new PalDBIndexMapIterator(this)
}

object PalDBIndexMap {
  // PalDB is not thread safe for parallel reads, even for different storages. It's necessary to lock it.
  private val PALDB_READER_LOCK = "READER_LOCK"

  /**
    * Returns the formatted filename for a partition file of PalDB IndexMap storing (name -> index) mapping
    * This method should be used consistently as a protocol to handle naming conventions
    *
    * @param partitionId The partition Id
    * @return The formatted filename
    */
  def partitionFilename(partitionId: Int, namespace: String = IndexMap.GLOBAL_NS): String =
    s"paldb-partition-$namespace-$partitionId.dat"

  /**
   *
   * @param indexMap
   */
  class PalDBIndexMapIterator(private val indexMap: PalDBIndexMap) extends Iterator[(String, Int)] {
    private var _iter: Iterator[JMap.Entry[Any, Any]] = _
    private var _currentItem: (String, Int) = _

    /**
     *
     * @return
     */
    override def hasNext: Boolean = {
      fetch()
      _currentItem != null
    }

    /**
     *
     * @return
     */
    override def next(): (String, Int) = {
      fetch()
      if (_currentItem == null) {
        throw new RuntimeException("No more element exists.")
      }

      val res = _currentItem
      _currentItem = null
      res
    }

    /**
     *
     */
    private def fetch(): Unit = {
      if (_iter == null) {
        val merged = indexMap
          ._storeReaders
          .foldLeft(List[JMap.Entry[Any, Any]]().toIterator){ (l, r) =>
            l ++ r.iterable().iterator().asScala
          }
        _iter = merged
      }

      while (_iter.hasNext && _currentItem == null) {
        val entry = _iter.next()
        (entry.getKey, entry.getValue) match {
          case (key: String, entryValue: Int) =>
            val i = indexMap.getPartitionId(key)
            val value = entryValue + indexMap._offsets(i)
            _currentItem = (key, value)
          case _ =>
        }
      }
    }
  }
}
