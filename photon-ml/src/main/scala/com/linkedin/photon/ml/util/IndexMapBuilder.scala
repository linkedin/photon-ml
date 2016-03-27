package com.linkedin.photon.ml.util

/**
  * Created by yizhou on 3/23/16.
  */
trait IndexMapBuilder {

  /**
    * Initialize an IndexMapBuilder, should be triggered as the 1st step of a builder
    *
    * @param outputDir The HDFS directory to store the built index map file
    * @param partitionId The partition id of current builder
    * @return the current builder
    */
  def init(outputDir: String, partitionId: Int): IndexMapBuilder

  /**
    * Close current builder
    */
  def close(): Unit

  /**
    * Automatically put a feature into the index map builder, using by default indexing logic handled by the builder
    *
    * @param name
    * @return
    */
  def putIfAbsent(name: String): IndexMapBuilder

  /**
    * Put a feature into map using a specific indexing rule
    *
    * @param name
    * @param idx
    * @return the current builder
    */
  def put(name: String, idx: Int): IndexMapBuilder
}
