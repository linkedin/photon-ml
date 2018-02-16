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
package com.linkedin.photon.ml.data

import org.apache.spark.sql.DataFrame

import com.linkedin.photon.ml.index.IndexMapLoader
import com.linkedin.photon.ml.io.FeatureShardConfiguration

/**
 * The DataReader interface. This interface should be implemented by readers for specific data formats.
 *
 * @param defaultFeatureColumn The default column to use for features
 */
abstract class DataReader(
    protected val defaultFeatureColumn: String = InputColumnsNames.FEATURES_DEFAULT.toString,
    protected val defaultIntercept: Boolean = true) {

  type InputColumnName = String
  type MergedColumnName = String

  /**
   * This map defines the "feature bags" or "feature shards".
   */
  private val defaultFeatureConfigMap = Map(
    defaultFeatureColumn -> FeatureShardConfiguration(Set(defaultFeatureColumn), defaultIntercept))

  /**
   * Reads the file at the given path into a DataFrame, assuming the default feature vector.
   *
   * @param path The path to the file or folder
   * @param numPartitions The minimum number of partitions. Spark is generally moving away from manually specifying
   *                      partition counts like this, in favor of inferring it. However, Photon currently still exposes
   *                      partition counts as a means for tuning job performance. The auto-inferred counts are usually
   *                      much lower than the necessary counts for Photon (especially GAME), so this caused a lot of
   *                      shuffling when repartitioning from the auto-partitioned data to the GAME data. We expose this
   *                      setting here to avoid the shuffling.
   * @return The loaded and transformed DataFrame
   */
  def read(path: String, numPartitions: Int): (DataFrame, IndexMapLoader) = read(Seq(path), numPartitions)

  /**
   * Reads the file at the given path into a DataFrame, assuming the default feature vector.
   *
   * @param path The path to the file or folder
   * @param numPartitions The minimum number of partitions. Spark is generally moving away from manually specifying
   *                      partition counts like this, in favor of inferring it. However, Photon currently still exposes
   *                      partition counts as a means for tuning job performance. The auto-inferred counts are usually
   *                      much lower than the necessary counts for Photon (especially GAME), so this caused a lot of
   *                      shuffling when repartitioning from the auto-partitioned data to the GAME data. We expose this
   *                      setting here to avoid the shuffling.
   * @param intercept Whether to add a feature representing the intercept to the feature vector
   * @return The loaded and transformed DataFrame
   */
  def read(path: String, numPartitions: Int, intercept: Boolean): (DataFrame, IndexMapLoader) =
    read(Seq(path), numPartitions, intercept)

  /**
   * Reads the file at the given path into a DataFrame, using the given index map for feature names.
   *
   * @param path The path to the file or folder
   * @param indexMapLoaders A map of index map loaders, containing one loader for each merged feature column
   * @param numPartitions The minimum number of partitions. Spark is generally moving away from manually specifying
   *                      partition counts like this, in favor of inferring it. However, Photon currently still exposes
   *                      partition counts as a means for tuning job performance. The auto-inferred counts are usually
   *                      much lower than the necessary counts for Photon (especially GAME), so this caused a lot of
   *                      shuffling when repartitioning from the auto-partitioned data to the GAME data. We expose this
   *                      setting here to avoid the shuffling.
   * @return The loaded and transformed DataFrame
   */
  def read(path: String, indexMapLoaders: Map[MergedColumnName, IndexMapLoader], numPartitions: Int): DataFrame =
    read(Seq(path), indexMapLoaders, numPartitions)

  /**
   * Reads the files at the given paths into a DataFrame, assuming the default feature vector.
   *
   * @param paths The paths to the files or folders
   * @param numPartitions The minimum number of partitions. Spark is generally moving away from manually specifying
   *                      partition counts like this, in favor of inferring it. However, Photon currently still exposes
   *                      partition counts as a means for tuning job performance. The auto-inferred counts are usually
   *                      much lower than the necessary counts for Photon (especially GAME), so this caused a lot of
   *                      shuffling when repartitioning from the auto-partitioned data to the GAME data. We expose this
   *                      setting here to avoid the shuffling.
   * @return The loaded and transformed DataFrame
   */
  def read(paths: Seq[String], numPartitions: Int): (DataFrame, IndexMapLoader) = {

    val (data, indexMapLoaders) = readMerged(paths, defaultFeatureConfigMap, numPartitions)

    (data, indexMapLoaders(defaultFeatureColumn))
  }

  /**
   * Reads the files at the given paths into a DataFrame, assuming the default feature vector.
   *
   * @param paths The paths to the files or folders
   * @param numPartitions The minimum number of partitions. Spark is generally moving away from manually specifying
   *                      partition counts like this, in favor of inferring it. However, Photon currently still exposes
   *                      partition counts as a means for tuning job performance. The auto-inferred counts are usually
   *                      much lower than the necessary counts for Photon (especially GAME), so this caused a lot of
   *                      shuffling when repartitioning from the auto-partitioned data to the GAME data. We expose this
   *                      setting here to avoid the shuffling.
   * @param intercept Whether to add a feature representing the intercept to the feature vector
   * @return The loaded and transformed DataFrame
   */
  def read(paths: Seq[String], numPartitions: Int, intercept: Boolean): (DataFrame, IndexMapLoader) = {

    val featureConfigMap = Map(defaultFeatureColumn -> FeatureShardConfiguration(Set(defaultFeatureColumn), intercept))
    val (data, indexMapLoaders) = readMerged(paths, featureConfigMap, numPartitions)

    (data, indexMapLoaders(defaultFeatureColumn))
  }

  /**
   * Reads the files at the given paths into a DataFrame, using the given index map for feature names.
   *
   * @param paths The paths to the files or folders
   * @param indexMapLoaders A map of index map loaders, containing one loader for each merged feature column
   * @param numPartitions The minimum number of partitions. Spark is generally moving away from manually specifying
   *                      partition counts like this, in favor of inferring it. However, Photon currently still exposes
   *                      partition counts as a means for tuning job performance. The auto-inferred counts are usually
   *                      much lower than the necessary counts for Photon (especially GAME), so this caused a lot of
   *                      shuffling when repartitioning from the auto-partitioned data to the GAME data. We expose this
   *                      setting here to avoid the shuffling.
   * @return The loaded and transformed DataFrame
   */
  def read(paths: Seq[String], indexMapLoaders: Map[MergedColumnName, IndexMapLoader], numPartitions: Int): DataFrame =
    readMerged(paths, indexMapLoaders, defaultFeatureConfigMap, numPartitions)

  /**
   * Reads the file at the given path into a DataFrame, using the given index map for feature names. Merges source
   * columns into combined feature vectors as specified by the featureColumnMap argument. Often features are joined from
   * different sources, and it can be more scalable to combine them into problem-specific feature vectors that can be
   * independently distributed.
   *
   * @param path The path to the file or folder
   * @param indexMapLoadersOpt An optional map of index map loaders, containing one loader for each merged feature
   *                           column
   * @param featureColumnConfigsMap A map that specifies how the feature columns should be merged. The keys specify the
   *                                name of the merged destination column, and the values are configs containing sets of
   *                                source columns to merge, e.g.:
   *
   *   Map("userFeatures" -> FeatureShardConfiguration(Set("profileFeatures", "titleFeatures")))
   *
   *                                This configuration merges the "profileFeatures" and "titleFeatures" columns into a
   *                                single column named "userFeatures".
   * @param numPartitions The minimum number of partitions. Spark is generally moving away from manually specifying
   *                      partition counts like this, in favor of inferring it. However, Photon currently still exposes
   *                      partition counts as a means for tuning job performance. The auto-inferred counts are usually
   *                      much lower than the necessary counts for Photon (especially GAME), so this caused a lot of
   *                      shuffling when repartitioning from the auto-partitioned data to the GAME data. We expose this
   *                      setting here to avoid the shuffling.
   * @return The loaded and transformed DataFrame
   */
  def readMerged(
      path: String,
      indexMapLoadersOpt: Option[Map[MergedColumnName, IndexMapLoader]],
      featureColumnConfigsMap: Map[MergedColumnName, FeatureShardConfiguration],
      numPartitions: Int): (DataFrame, Map[MergedColumnName, IndexMapLoader]) =
    readMerged(Seq(path), indexMapLoadersOpt, featureColumnConfigsMap, numPartitions)

  /**
   * Reads the file at the given path into a DataFrame, using the given index map for feature names. Merges source
   * columns into combined feature vectors as specified by the featureColumnMap argument. Often features are joined from
   * different sources, and it can be more scalable to combine them into problem-specific feature vectors that can be
   * independently distributed.
   *
   * @param paths The path to the files or folders
   * @param indexMapLoadersOpt An optional map of index map loaders, containing one loader for each merged feature
   *                           column
   * @param featureColumnConfigsMap A map that specifies how the feature columns should be merged. The keys specify the
   *                                name of the merged destination column, and the values are configs containing sets of
   *                                source columns to merge, e.g.:
   *
   *   Map("userFeatures" -> FeatureShardConfiguration(Set("profileFeatures", "titleFeatures")))
   *
   *                                This configuration merges the "profileFeatures" and "titleFeatures" columns into a
   *                                single column named "userFeatures".
   * @param numPartitions The minimum number of partitions. Spark is generally moving away from manually specifying
   *                      partition counts like this, in favor of inferring it. However, Photon currently still exposes
   *                      partition counts as a means for tuning job performance. The auto-inferred counts are usually
   *                      much lower than the necessary counts for Photon (especially GAME), so this caused a lot of
   *                      shuffling when repartitioning from the auto-partitioned data to the GAME data. We expose this
   *                      setting here to avoid the shuffling.
   * @return The loaded and transformed DataFrame
   */
  def readMerged(
      paths: Seq[String],
      indexMapLoadersOpt: Option[Map[MergedColumnName, IndexMapLoader]],
      featureColumnConfigsMap: Map[MergedColumnName, FeatureShardConfiguration],
      numPartitions: Int): (DataFrame, Map[MergedColumnName, IndexMapLoader]) =
    indexMapLoadersOpt match {
      case Some(indexMapLoaders) =>
        (readMerged(paths, indexMapLoaders, featureColumnConfigsMap, numPartitions), indexMapLoaders)

      case None =>
        readMerged(paths, featureColumnConfigsMap, numPartitions)
    }

  /**
   * Reads the file at the given path into a DataFrame, generating a default index map for feature names. Merges
   * source columns into combined feature vectors as specified by the featureColumnMap argument. Often features are
   * joined from different sources, and it can be more scalable to combine them into problem-specific feature vectors
   * that can be independently distributed.
   *
   * @param path The path to the file or folder
   * @param featureColumnConfigsMap A map that specifies how the feature columns should be merged. The keys specify the
   *                                name of the merged destination column, and the values are configs containing sets of
   *                                source columns to merge, e.g.:
   *
   *   Map("userFeatures" -> FeatureShardConfiguration(Set("profileFeatures", "titleFeatures")))
   *
   *                                This configuration merges the "profileFeatures" and "titleFeatures" columns into a
   *                                single column named "userFeatures".
   * @param numPartitions The minimum number of partitions. Spark is generally moving away from manually specifying
   *                      partition counts like this, in favor of inferring it. However, Photon currently still exposes
   *                      partition counts as a means for tuning job performance. The auto-inferred counts are usually
   *                      much lower than the necessary counts for Photon (especially GAME), so this caused a lot of
   *                      shuffling when repartitioning from the auto-partitioned data to the GAME data. We expose this
   *                      setting here to avoid the shuffling.
   * @return The loaded and transformed DataFrame
   */
  def readMerged(
      path: String,
      featureColumnConfigsMap: Map[MergedColumnName, FeatureShardConfiguration],
      numPartitions: Int): (DataFrame, Map[MergedColumnName, IndexMapLoader]) =
    readMerged(Seq(path), featureColumnConfigsMap, numPartitions)

  /**
   * Reads the file at the given path into a DataFrame, using the given index map for feature names. Merges source
   * columns into combined feature vectors as specified by the featureColumnMap argument. Often features are joined from
   * different sources, and it can be more scalable to combine them into problem-specific feature vectors that can be
   * independently distributed.
   *
   * @param path The path to the file or folder
   * @param indexMapLoaders A map of index map loaders, containing one loader for each merged feature column
   * @param featureColumnConfigsMap A map that specifies how the feature columns should be merged. The keys specify the
   *                                name of the merged destination column, and the values are configs containing sets of
   *                                source columns to merge, e.g.:
   *
   *   Map("userFeatures" -> FeatureShardConfiguration(Set("profileFeatures", "titleFeatures")))
   *
   *                                This configuration merges the "profileFeatures" and "titleFeatures" columns into a
   *                                single column named "userFeatures".
   * @param numPartitions The minimum number of partitions. Spark is generally moving away from manually specifying
   *                      partition counts like this, in favor of inferring it. However, Photon currently still exposes
   *                      partition counts as a means for tuning job performance. The auto-inferred counts are usually
   *                      much lower than the necessary counts for Photon (especially GAME), so this caused a lot of
   *                      shuffling when repartitioning from the auto-partitioned data to the GAME data. We expose this
   *                      setting here to avoid the shuffling.
   * @return The loaded and transformed DataFrame
   */
  def readMerged(
      path: String,
      indexMapLoaders: Map[MergedColumnName, IndexMapLoader],
      featureColumnConfigsMap: Map[MergedColumnName, FeatureShardConfiguration],
      numPartitions: Int): DataFrame =
    readMerged(Seq(path), indexMapLoaders, featureColumnConfigsMap, numPartitions)

  /**
   * Reads the files at the given paths into a DataFrame, generating a default index map for feature names. Merges
   * source columns into combined feature vectors as specified by the featureColumnMap argument. Often features are
   * joined from different sources, and it can be more scalable to combine them into problem-specific feature vectors
   * that can be independently distributed.
   *
   * @param paths The path to the files or folders
   * @param featureColumnConfigsMap A map that specifies how the feature columns should be merged. The keys specify the
   *                                name of the merged destination column, and the values are configs containing sets of
   *                                source columns to merge, e.g.:
   *
   *   Map("userFeatures" -> FeatureShardConfiguration(Set("profileFeatures", "titleFeatures")))
   *
   *                                This configuration merges the "profileFeatures" and "titleFeatures" columns into a
   *                                single column named "userFeatures".
   * @param numPartitions The minimum number of partitions. Spark is generally moving away from manually specifying
   *                      partition counts like this, in favor of inferring it. However, Photon currently still exposes
   *                      partition counts as a means for tuning job performance. The auto-inferred counts are usually
   *                      much lower than the necessary counts for Photon (especially GAME), so this caused a lot of
   *                      shuffling when repartitioning from the auto-partitioned data to the GAME data. We expose this
   *                      setting here to avoid the shuffling.
   * @return The loaded and transformed DataFrame
   */
  def readMerged(
      paths: Seq[String],
      featureColumnConfigsMap: Map[MergedColumnName, FeatureShardConfiguration],
      numPartitions: Int): (DataFrame, Map[MergedColumnName, IndexMapLoader])

  /**
   * Reads the files at the given paths into a DataFrame, using the given index map for feature names. Merges source
   * columns into combined feature vectors as specified by the featureColumnMap argument. Often features are joined from
   * different sources, and it can be more scalable to combine them into problem-specific feature vectors that can be
   * independently distributed.
   *
   * @param paths The path to the files or folders
   * @param indexMapLoaders A map of index map loaders, containing one loader for each merged feature column
   * @param featureColumnConfigsMap A map that specifies how the feature columns should be merged. The keys specify the
   *                                name of the merged destination column, and the values are configs containing sets of
   *                                source columns to merge, e.g.:
   *
   *   Map("userFeatures" -> FeatureShardConfiguration(Set("profileFeatures", "titleFeatures")))
   *
   *                                This configuration merges the "profileFeatures" and "titleFeatures" columns into a
   *                                single column named "userFeatures".
   * @param numPartitions The minimum number of partitions. Spark is generally moving away from manually specifying
   *                      partition counts like this, in favor of inferring it. However, Photon currently still exposes
   *                      partition counts as a means for tuning job performance. The auto-inferred counts are usually
   *                      much lower than the necessary counts for Photon (especially GAME), so this caused a lot of
   *                      shuffling when repartitioning from the auto-partitioned data to the GAME data. We expose this
   *                      setting here to avoid the shuffling.
   * @return The loaded and transformed DataFrame
   */
  def readMerged(
      paths: Seq[String],
      indexMapLoaders: Map[MergedColumnName, IndexMapLoader],
      featureColumnConfigsMap: Map[MergedColumnName, FeatureShardConfiguration],
      numPartitions: Int): DataFrame
}
