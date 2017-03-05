/*
 * Copyright 2017 LinkedIn Corp. All rights reserved.
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

import scala.language.higherKinds

/**
 * This object contains a few implicits that we used to make the syntax cleaner, by removing unessential details.
 * Note that as a team, we tend to avoid implicits when they make tracing harder ("invisible" implicits). In that
 * respect, the implicits here have specific names ("tap") that hopefully help in tracing.
 */
protected[ml] object Implicits {
  /**
   * This is convenient to write e.g. (so the intent is cleaner syntax):
   *
   *   calculateStatistics(trainingData, indexMapLoaders)
   *     .tap {
   *       (featureShardId, shardSummary) =>
   *         val outputDir = summarizationOutputDir + "/" + featureShardId
   *         val indexMap = indexMapLoaders(featureShardId).indexMapForDriver()
   *         IOUtils.writeBasicStatistics(sc, shardSummary, outputDir, indexMap)
   *       }
   *
   * Without having to create a val, then calling map on the val, then repeating the val name at the end
   * to do something with it. In essence, we append a block of side-effect'ing code to a chain of operations
   * on a Map.
   *
   * @param m A Map to tap
   * @tparam K The type of the keys in Map m
   * @tparam V The type of the values in Map m
   */
  implicit class TapMap[K, V](m: Map[K, V]) {

    def tap(f: ((K, V)) => Unit): Map[K, V] = { m.foreach(f); m }
    def tapKeys(f: K => Unit): Map[K, V] = { m.keysIterator.foreach(f); m }
    def tapValues(f: V => Unit): Map[K, V] = { m.valuesIterator.foreach(f); m }
  }

  /**
   * Tap for Set, List, Iterable ... Same idea as tap for Map, same usage pattern: append some side effect in
   * a chain of operations on a Set, List ...
   *
   * @param c A Collection to iterate on
   * @param ev An implicit to make sure the collection is TraversableOnce
   * @tparam Coll The type of the Collection
   * @tparam T The type of the elements in the Collection
   */
  implicit class TapTraversable[Coll[_], T](c: Coll[T])(implicit ev: Coll[T] <:< TraversableOnce[T]) {
    def tap(f: T => Unit): Coll[T] = { ev(c).foreach(f); c }
  }

  /**
   * Tap for Array, which is not quite Iterable. See documentation for TapMap.
   *
   * @param a The Array to tap
   * @tparam T The type of whatever is in the Array
   */
  implicit class TapArray[T](a: Array[T]) {
    def tap(f: T => Unit): Array[T] = { a.foreach(f); a }
  }

  /**
   * Tap for Options. See documentation for TapMap.
   *
   * @param o An Option to tap
   * @tparam T The type of the thing contained in the Option
   */
  implicit class TapOption[T](o: Option[T]) {
    def tap(f: T => Unit): Option[T] = { o.foreach(f); o }
  }

  /**
   * Extract a value from a Map 'through' an Option. This allows to write simpler code, e.g.:
   *
   *   normalizationContexts.extractOrElse(shardId)(defaultNormalizationContext)
   *
   * given:
   *
   *   normalizationContexts: Option[Map[String, NormalizationContext]]
   *
   * When an Option contains a Map[K, V], extract V at K key from Map if Option isDefined, else apply f that
   * returns a default V.
   *
   * @note throws an exception if the Option isDefined but key is not found. The behavior of the Map is not changed,
   * the default provided is used only in case the Option itself isEmpty.
   *
   * @param o An Option that contains a Map[K, V]
   * @tparam K The type of the keys in the Map
   * @tparam V The type of the values in the Map
   */
  implicit class ExtractOrElse[K, V](o: Option[Map[K, V]]) {
    def extractOrElse(key: K)(f: => V): V = { if (o.isDefined) o.get(key) else f }
  }
}
