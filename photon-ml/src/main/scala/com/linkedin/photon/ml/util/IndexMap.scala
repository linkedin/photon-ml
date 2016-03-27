package com.linkedin.photon.ml.util

/**
  * The trait defines the methods supposed should be supported by an index map
  *
  * @author yizhou
  */
trait IndexMap extends Map[String, Int] with java.io.Serializable {

  /**
    * Given an index, reversely track down the corresponding feature name
    *
    * @param idx the feature index
    * @return the feature name, return null if not found
    */
  def getFeatureName(idx: Int): String

  /**
    * Given a feature string, return the index
    *
    * @param name the feature name
    * @return the feature index, return IndexMap.NULL_KEY if not found
    */
  def getIndex(name: String): Int
}

object IndexMap {
  // The key to indicate a feature is not existing in the map
  val NULL_KEY:Int = -1
}
