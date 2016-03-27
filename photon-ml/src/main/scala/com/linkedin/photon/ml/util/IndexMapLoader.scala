package com.linkedin.photon.ml.util


import com.linkedin.photon.ml.Params
import org.apache.spark.SparkContext


/**
  * This class should provide a universal approach of loading an IndexMap.
  *
  * To access an IndexMap within RDD operations, directly referring to an object inside Driver is inefficient.
  * The driver will try to serialize the entire object onto RDDs. This trait is targeted to solve such problem that
  * regardless of the implementation of different IndexMaps, we should provide a consistent way of loading them
  * anywhere.
  *
  * Also its up to IndexMap to decide whether each time, it should provide a reusable instance or create a new one.
  *
  * @author yizhou
  */
trait IndexMapLoader extends java.io.Serializable {

  /**
    * Prepare the loader, should be called at the very beginning
    *
    * @param sc the SparkContext
    * @param params the parameters object
    */
  def prepare(sc: SparkContext, params: Params): Unit

  /**
    * Should be called in driver
    *
    * @return the loaded IndexMap for driver
    */
  def indexMapForDriver(): IndexMap

  /**
    * Should be called inside RDD operations
    *
    * @return the loaded IndexMap for RDDs
    */
  def indexMapForRDD(): IndexMap
}
