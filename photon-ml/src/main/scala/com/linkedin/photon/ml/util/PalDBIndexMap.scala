package com.linkedin.photon.ml.util

import com.linkedin.paldb.api.{Configuration, PalDB, StoreReader}
import org.apache.spark.{SparkFiles, HashPartitioner}


/**
  * An off heap index map implementation using PalDB
  *
  * @author yizhou
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

  def load(storePath: String, partitionsNum: Int): PalDBIndexMap = {
    _storeReaders = new Array[StoreReader](partitionsNum)
    _offsets = new Array[Int](partitionsNum)

    _partitionsNum = partitionsNum
    _partitioner = new HashPartitioner(_partitionsNum)

    for (i <- 0 until partitionsNum) {
      val config = new Configuration()
      config.set(Configuration.CACHE_ENABLED, "true")
      // Allow 200MB in-memory cache in total
      config.set(Configuration.CACHE_BYTES, String.valueOf(209715200L / _partitionsNum))

      // TODO: because we also store reverse mapping
      _offsets(i) = _size / 2
      val filename = getPartitionFilename(i)
      val storeFile = new java.io.File(SparkFiles.get(filename))
      _storeReaders(i) = PalDB.createReader(storeFile, config)
      _size += _storeReaders(i).size().asInstanceOf[Number].intValue()
    }

    this
  }

  override def isEmpty(): Boolean = _storeReaders == null

  override def size(): Int = _size / 2

  override def getFeatureName(idx: Int): String = {
    // throw new RuntimeException("Reverse query is not supported yet")
    var i = 0
    for (i <- 0 until _partitionsNum) {
      val k = idx - _offsets(i)
      val v = _storeReaders(i).get(k).asInstanceOf[String]
      if (v != null) {
        return v
      }
    }

    null
  }

  override def getIndex(name: String): Int = {
    val i = _partitioner.getPartition(name)
    val map = _storeReaders(i)
    // Note: very important to cast to java.lang.Object first; if directly casting to int,
    // null will be cast to 0 by Scala
    val idx = map.get(name).asInstanceOf[java.lang.Object]

    if (idx == null) -1 else idx.asInstanceOf[Int] + _offsets(i)
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

  /**
    * Returns the formatted filename for a partitioncular partition file of PalDB IndexMap.
    * This method should be used consistently as a protocol to handle naming conventions
    *
    * @param partitionId the partition Id
    * @return the formatted filename
    */
  def getPartitionFilename(partitionId: Int): String = s"paldb-partition-${partitionId}.dat"

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
        throw new RuntimeException("No more element exisits.")
      }

      val res = _currentItem
      _currentItem = null
      res
    }

    private def fetch(): Unit = {
        while(_currentItem == null && ((_storeIterator != null && _storeIterator.hasNext) || _i < map._partitionsNum - 1)) {
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
