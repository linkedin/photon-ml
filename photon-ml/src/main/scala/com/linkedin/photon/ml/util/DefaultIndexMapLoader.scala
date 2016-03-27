package com.linkedin.photon.ml.util


import com.linkedin.photon.ml.Params
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast


/**
  * A loader that provides instances of DefaultIndexMap
  *
  * @author yizhou
  */
class DefaultIndexMapLoader(featureNameToIdMap: Map[String, Int]) extends IndexMapLoader {
  @transient
  private var _indexMap: IndexMap = null

  private var _mapBroadCaster: Broadcast[Map[String, Int]] = null

  /**
    * Prepare a loader, should be called early before anything
    */
  override def prepare(sc: SparkContext, params: Params): Unit = {
    // do nothing
    _indexMap = new DefaultIndexMap(featureNameToIdMap)
    _mapBroadCaster = sc.broadcast(featureNameToIdMap)
  }

  /**
    * Should be called in driver
    *
    * @return a new IndexMap
    */
  override def indexMapForDriver(): IndexMap = _indexMap

  /**
    * Should be called inside RDD operations
    *
    * @return a new IndexMap
    */
  override def indexMapForRDD(): IndexMap = new DefaultIndexMap(_mapBroadCaster.value)
}
