package com.linkedin.photon.ml.util

/**
  * Use the default system HashMap to construct an index map, highly inefficient in terms of memory usage; but easier to
  * handle. Recommended for small feature space cases (<= 200k).
  *
  * @author yizhou
  */
class DefaultIndexMap(val featureNameToIdMap: Map[String, Int]) extends IndexMap {
  private var _idToNameMap: Map[Int, String] = null

  override def getFeatureName(idx: Int): String = {
    if (_idToNameMap == null) {
      _idToNameMap = featureNameToIdMap.map{case (k, v) => (v, k)}
    }

    _idToNameMap.getOrElse(idx, null)
  }

  override def getIndex(name: String): Int = featureNameToIdMap.getOrElse(name, IndexMap.NULL_KEY)

  override def +[B1 >: Int](kv: (String, B1)): Map[String, B1] = featureNameToIdMap.+(kv)

  override def get(key: String): Option[Int] = featureNameToIdMap.get(key)

  override def iterator: Iterator[(String, Int)] = featureNameToIdMap.iterator

  override def -(key: String): Map[String, Int] = featureNameToIdMap.-(key)
}
