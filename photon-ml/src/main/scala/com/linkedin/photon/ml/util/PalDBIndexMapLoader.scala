package com.linkedin.photon.ml.util


import com.linkedin.photon.ml.Params
import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext


/**
  * A PalDBIndexMap loader
  *
  * @author yizhou
  */
class PalDBIndexMapLoader extends IndexMapLoader {
  private var _storeDir: String = null
  private var _numPartitions: Int = 0

  override def prepare(sc: SparkContext, params: Params): Unit = {
    if (!params.offHeapIndexMapDir.isEmpty && params.offHeapIndexMapNumPartitions != 0) {
      _storeDir = params.offHeapIndexMapDir
      _numPartitions = params.offHeapIndexMapNumPartitions
      (0 until params.offHeapIndexMapNumPartitions).foreach(i =>
        sc.addFile(new Path(_storeDir, PalDBIndexMap.getPartitionFilename(i)).toUri().toString())
      )
    }
  }

  override def indexMapForDriver(): IndexMap = new PalDBIndexMap().load(_storeDir, _numPartitions)

  /**
    * Should be called inside RDD operations
    *
    * @return the loaded IndexMap for RDDs
    */
  override def indexMapForRDD(): IndexMap = new PalDBIndexMap().load(_storeDir, _numPartitions)
}
