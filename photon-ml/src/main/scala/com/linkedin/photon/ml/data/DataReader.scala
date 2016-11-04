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
package com.linkedin.photon.ml.data

import com.linkedin.photon.ml.util.IndexMapLoader

import org.apache.spark.sql.DataFrame

/**
 * The DataReader interface. This interface should be implemented by readers for specific data formats.
 *
 * @param sc the Spark context
 * @param defaultFeatureColumn the default column to use for features
 */
abstract class DataReader(protected val defaultFeatureColumn: String = "features") {

  val defaultFeatureColumnMap = Map(defaultFeatureColumn -> Set(defaultFeatureColumn))

  /**
   * Reads the file at the given path into a DataFrame, generating a default index map for feature names. Merges source
   * columns into combined feature vectors as specified by the featureColumnMap argument. Often features are joined from
   * different sources, and it can be more scalable to combine them into problem-specific feature vectors that can be
   * independently distributed.
   *
   * @param path the path to the file or folder
   * @param featureColumnMap a map that specifies how the feature columns should be merged. The keys specify the name
   *   of the merged destination column, and the values are sets of source columns to merge, e.g.:
   *
   *     Map("userFeatures" -> Set("profileFeatures", "titleFeatures"))
   *
   *   This configuration merges the "profileFeatures" and "titleFeatures" columns into a single column named
   *   "userFeatures".
   * @return the loaded and transformed DataFrame
   */
  def readMerged(
      path: String,
      featureColumnMap: Map[String, Set[String]]): (DataFrame, Map[String, IndexMapLoader])

  /**
   * Reads the file at the given path into a DataFrame, using the given index map for feature names. Merges source
   * columns into combined feature vectors as specified by the featureColumnMap argument. Often features are joined from
   * different sources, and it can be more scalable to combine them into problem-specific feature vectors that can be
   * independently distributed.
   *
   * @param path the path to the file or folder
   * @param indexMapLoaders a map of index map loaders, containing one loader for each merged feature column
   * @param featureColumnMap a map that specifies how the feature columns should be merged. The keys specify the name
   *   of the merged destination column, and the values are sets of source columns to merge, e.g.:
   *
   *     Map("userFeatures" -> Set("profileFeatures", "titleFeatures"))
   *
   *   This configuration merges the "profileFeatures" and "titleFeatures" columns into a single column named
   *   "userFeatures".
   * @return the loaded and transformed DataFrame
   */
  def readMerged(
      path: String,
      indexMapLoaders: Map[String, IndexMapLoader],
      featureColumnMap: Map[String, Set[String]]): DataFrame

  /**
   * Reads the file at the given path into a DataFrame, assuming the default feature vector
   *
   * @param path the path to the file or folder
   * @return the loaded and transformed DataFrame
   */
  def read(path: String): (DataFrame, IndexMapLoader) = {
    val (data, indexMapLoaders) = readMerged(path, defaultFeatureColumnMap)
    (data, indexMapLoaders(defaultFeatureColumn))
  }

  /**
   * Reads the file at the given path into a DataFrame, using the given index map for feature names
   *
   * @param path the path to the file or folder
   * @param indexMapLoaders a map of index map loaders, containing one loader for each merged feature column
   * @return the loaded and transformed DataFrame
   */
  def read(path: String, indexMapLoaders: Map[String, IndexMapLoader]): DataFrame =
    readMerged(path, indexMapLoaders, defaultFeatureColumnMap)

  /**
   * Reads the files at the given paths into a DataFrame, generating a default index map for feature names. Merges
   * source columns into combined feature vectors as specified by the featureColumnMap argument. Often features are
   * joined from different sources, and it can be more scalable to combine them into problem-specific feature vectors
   * that can be independently distributed.
   *
   * @param paths the paths to the files or folders
   * @param featureColumnMap a map that specifies how the feature columns should be merged. The keys specify the name
   *   of the merged destination column, and the values are sets of source columns to merge, e.g.:
   *
   *     Map("userFeatures" -> Set("profileFeatures", "titleFeatures"))
   *
   *   This configuration merges the "profileFeatures" and "titleFeatures" columns into a single column named
   *   "userFeatures".
   * @return the loaded and transformed DataFrame
   */
  def readMerged(
      paths: Seq[String],
      featureColumnMap: Map[String, Set[String]]): (DataFrame, Map[String, IndexMapLoader]) = {

    val (first, indexMapLoaders) = readMerged(paths.head, featureColumnMap)
    val df = paths.tail.foldLeft(first) { case (acc: DataFrame, path: String) =>
      // Note: This isn't quite right: in this case where we haven't passed in a pre-generated index map, we're
      // generating a default map for only the first file. This is because we currently don't have a way to merge index
      // maps. This shouldn't be a huge problem, since this code path is meant mosly for exploratory purposes and not
      // production code, where the feature maps will have been generated ahead of time. It should be addressed
      // eventually, however.
      val curr = readMerged(path, indexMapLoaders, featureColumnMap)
      acc.unionAll(curr)
    }

    (df, indexMapLoaders)
  }

  /**
   * Reads the files at the given paths into a DataFrame, using the given index map for feature names. Merges source
   * columns into combined feature vectors as specified by the featureColumnMap argument. Often features are joined from
   * different sources, and it can be more scalable to combine them into problem-specific feature vectors that can be
   * independently distributed.
   *
   * @param paths the paths to the files or folders
   * @param indexMapLoaders a map of index map loaders, containing one loader for each merged feature column
   * @param featureColumnMap a map that specifies how the feature columns should be merged. The keys specify the name
   *   of the merged destination column, and the values are sets of source columns to merge, e.g.:
   *
   *     Map("userFeatures" -> Set("profileFeatures", "titleFeatures"))
   *
   *   This configuration merges the "profileFeatures" and "titleFeatures" columns into a single column named
   *   "userFeatures".
   * @return the loaded and transformed DataFrame
   */
  def readMerged(
      paths: Seq[String],
      indexMapLoaders: Map[String, IndexMapLoader],
      featureColumnMap: Map[String, Set[String]]): DataFrame = {

    val first = readMerged(paths.head, indexMapLoaders, featureColumnMap)
    val df = paths.tail.foldLeft(first) { case (acc: DataFrame, path: String) =>
      acc.unionAll(readMerged(path, indexMapLoaders, featureColumnMap))
    }

    df
  }

  /**
   * Reads the files at the given paths into a DataFrame, assuming the default feature vector
   *
   * @param paths the paths to the files or folders
   * @return the loaded and transformed DataFrame
   */
  def read(paths: Seq[String]): (DataFrame, IndexMapLoader) = {
    val (data, indexMapLoaders) = readMerged(paths, defaultFeatureColumnMap)
    (data, indexMapLoaders(defaultFeatureColumn))
  }

  /**
   * Reads the files at the given paths into a DataFrame, using the given index map for feature names
   *
   * @param paths the paths to the files or folders
   * @param indexMapLoaders a map of index map loaders, containing one loader for each merged feature column
   * @return the loaded and transformed DataFrame
   */
  def read(paths: Seq[String], indexMapLoaders: Map[String, IndexMapLoader]): DataFrame =
    readMerged(paths, indexMapLoaders, defaultFeatureColumnMap)

}
