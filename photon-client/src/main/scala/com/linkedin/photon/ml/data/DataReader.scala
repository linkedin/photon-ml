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
   * Reads the file at the given path into a [[DataFrame]], assuming the default feature vector.
   *
   * @param path The path to the file or folder
   * @param numPartitionsOpt An optional minimum number of partitions for the [[DataFrame]]. If no minimum is provided,
   *                         the default parallelism is used. Spark is generally moving away from manually specifying
   *                         partition counts like this, in favor of inferring it. However, Photon still exposes
   *                         partition counts as a means for tuning job performance. The auto-inferred counts are
   *                         usually much lower than the necessary counts for Photon (especially GAME). This causes a
   *                         lot of shuffling when repartitioning from the auto-partitioned data to the processed GAME
   *                         data. This setting is exposed to allow tuning which can avoid the shuffling.
   * @return The loaded and transformed DataFrame
   */
  def read(path: String, numPartitionsOpt: Option[Int]): (DataFrame, IndexMapLoader) = read(Seq(path), numPartitionsOpt)

  /**
   * Reads the file at the given path into a DataFrame, assuming the default feature vector.
   *
   * @param path The path to the file or folder
   * @param numPartitionsOpt An optional minimum number of partitions for the [[DataFrame]]. If no minimum is provided,
   *                         the default parallelism is used. Spark is generally moving away from manually specifying
   *                         partition counts like this, in favor of inferring it. However, Photon still exposes
   *                         partition counts as a means for tuning job performance. The auto-inferred counts are
   *                         usually much lower than the necessary counts for Photon (especially GAME). This causes a
   *                         lot of shuffling when repartitioning from the auto-partitioned data to the processed GAME
   *                         data. This setting is exposed to allow tuning which can avoid the shuffling.
   * @param intercept Whether to add a feature representing the intercept to the feature vector
   * @return The loaded and transformed DataFrame
   */
  def read(path: String, numPartitionsOpt: Option[Int], intercept: Boolean): (DataFrame, IndexMapLoader) =
    read(Seq(path), numPartitionsOpt, intercept)

  /**
   * Reads the file at the given path into a DataFrame, using the given index map for feature names.
   *
   * @param path The path to the file or folder
   * @param indexMapLoaders A map of index map loaders, containing one loader for each merged feature column
   * @param numPartitionsOpt An optional minimum number of partitions for the [[DataFrame]]. If no minimum is provided,
   *                         the default parallelism is used. Spark is generally moving away from manually specifying
   *                         partition counts like this, in favor of inferring it. However, Photon still exposes
   *                         partition counts as a means for tuning job performance. The auto-inferred counts are
   *                         usually much lower than the necessary counts for Photon (especially GAME). This causes a
   *                         lot of shuffling when repartitioning from the auto-partitioned data to the processed GAME
   *                         data. This setting is exposed to allow tuning which can avoid the shuffling.
   * @return The loaded and transformed DataFrame
   */
  def read(
      path: String,
      indexMapLoaders: Map[MergedColumnName, IndexMapLoader],
      numPartitionsOpt: Option[Int]): DataFrame =
    read(Seq(path), indexMapLoaders, numPartitionsOpt)

  /**
   * Reads the files at the given paths into a DataFrame, assuming the default feature vector.
   *
   * @param paths The paths to the files or folders
   * @param numPartitionsOpt An optional minimum number of partitions for the [[DataFrame]]. If no minimum is provided,
   *                         the default parallelism is used. Spark is generally moving away from manually specifying
   *                         partition counts like this, in favor of inferring it. However, Photon still exposes
   *                         partition counts as a means for tuning job performance. The auto-inferred counts are
   *                         usually much lower than the necessary counts for Photon (especially GAME). This causes a
   *                         lot of shuffling when repartitioning from the auto-partitioned data to the processed GAME
   *                         data. This setting is exposed to allow tuning which can avoid the shuffling.
   * @return The loaded and transformed DataFrame
   */
  def read(paths: Seq[String], numPartitionsOpt: Option[Int]): (DataFrame, IndexMapLoader) = {

    val (data, indexMapLoaders) = readMerged(paths, defaultFeatureConfigMap, numPartitionsOpt)

    (data, indexMapLoaders(defaultFeatureColumn))
  }

  /**
   * Reads the files at the given paths into a DataFrame, assuming the default feature vector.
   *
   * @param paths The paths to the files or folders
   * @param numPartitionsOpt An optional minimum number of partitions for the [[DataFrame]]. If no minimum is provided,
   *                         the default parallelism is used. Spark is generally moving away from manually specifying
   *                         partition counts like this, in favor of inferring it. However, Photon still exposes
   *                         partition counts as a means for tuning job performance. The auto-inferred counts are
   *                         usually much lower than the necessary counts for Photon (especially GAME). This causes a
   *                         lot of shuffling when repartitioning from the auto-partitioned data to the processed GAME
   *                         data. This setting is exposed to allow tuning which can avoid the shuffling.
   * @param intercept Whether to add a feature representing the intercept to the feature vector
   * @return The loaded and transformed DataFrame
   */
  def read(paths: Seq[String], numPartitionsOpt: Option[Int], intercept: Boolean): (DataFrame, IndexMapLoader) = {

    val featureConfigMap = Map(defaultFeatureColumn -> FeatureShardConfiguration(Set(defaultFeatureColumn), intercept))
    val (data, indexMapLoaders) = readMerged(paths, featureConfigMap, numPartitionsOpt)

    (data, indexMapLoaders(defaultFeatureColumn))
  }

  /**
   * Reads the files at the given paths into a DataFrame, using the given index map for feature names.
   *
   * @param paths The paths to the files or folders
   * @param indexMapLoaders A map of index map loaders, containing one loader for each merged feature column
   * @param numPartitionsOpt An optional minimum number of partitions for the [[DataFrame]]. If no minimum is provided,
   *                         the default parallelism is used. Spark is generally moving away from manually specifying
   *                         partition counts like this, in favor of inferring it. However, Photon still exposes
   *                         partition counts as a means for tuning job performance. The auto-inferred counts are
   *                         usually much lower than the necessary counts for Photon (especially GAME). This causes a
   *                         lot of shuffling when repartitioning from the auto-partitioned data to the processed GAME
   *                         data. This setting is exposed to allow tuning which can avoid the shuffling.
   * @return The loaded and transformed DataFrame
   */
  def read(
      paths: Seq[String],
      indexMapLoaders: Map[MergedColumnName, IndexMapLoader],
      numPartitionsOpt: Option[Int]): DataFrame =
    readMerged(paths, indexMapLoaders, defaultFeatureConfigMap, numPartitionsOpt)

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
   * @param numPartitionsOpt An optional minimum number of partitions for the [[DataFrame]]. If no minimum is provided,
   *                         the default parallelism is used. Spark is generally moving away from manually specifying
   *                         partition counts like this, in favor of inferring it. However, Photon still exposes
   *                         partition counts as a means for tuning job performance. The auto-inferred counts are
   *                         usually much lower than the necessary counts for Photon (especially GAME). This causes a
   *                         lot of shuffling when repartitioning from the auto-partitioned data to the processed GAME
   *                         data. This setting is exposed to allow tuning which can avoid the shuffling.
   * @return The loaded and transformed DataFrame
   */
  def readMerged(
      path: String,
      indexMapLoadersOpt: Option[Map[MergedColumnName, IndexMapLoader]],
      featureColumnConfigsMap: Map[MergedColumnName, FeatureShardConfiguration],
      numPartitionsOpt: Option[Int]): (DataFrame, Map[MergedColumnName, IndexMapLoader]) =
    readMerged(Seq(path), indexMapLoadersOpt, featureColumnConfigsMap, numPartitionsOpt)

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
   * @param numPartitionsOpt An optional minimum number of partitions for the [[DataFrame]]. If no minimum is provided,
   *                         the default parallelism is used. Spark is generally moving away from manually specifying
   *                         partition counts like this, in favor of inferring it. However, Photon still exposes
   *                         partition counts as a means for tuning job performance. The auto-inferred counts are
   *                         usually much lower than the necessary counts for Photon (especially GAME). This causes a
   *                         lot of shuffling when repartitioning from the auto-partitioned data to the processed GAME
   *                         data. This setting is exposed to allow tuning which can avoid the shuffling.
   * @return The loaded and transformed DataFrame
   */
  def readMerged(
      paths: Seq[String],
      indexMapLoadersOpt: Option[Map[MergedColumnName, IndexMapLoader]],
      featureColumnConfigsMap: Map[MergedColumnName, FeatureShardConfiguration],
      numPartitionsOpt: Option[Int]): (DataFrame, Map[MergedColumnName, IndexMapLoader]) =
    indexMapLoadersOpt match {
      case Some(indexMapLoaders) =>
        (readMerged(paths, indexMapLoaders, featureColumnConfigsMap, numPartitionsOpt), indexMapLoaders)

      case None =>
        readMerged(paths, featureColumnConfigsMap, numPartitionsOpt)
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
   * @param numPartitionsOpt An optional minimum number of partitions for the [[DataFrame]]. If no minimum is provided,
   *                         the default parallelism is used. Spark is generally moving away from manually specifying
   *                         partition counts like this, in favor of inferring it. However, Photon still exposes
   *                         partition counts as a means for tuning job performance. The auto-inferred counts are
   *                         usually much lower than the necessary counts for Photon (especially GAME). This causes a
   *                         lot of shuffling when repartitioning from the auto-partitioned data to the processed GAME
   *                         data. This setting is exposed to allow tuning which can avoid the shuffling.
   * @return The loaded and transformed DataFrame
   */
  def readMerged(
      path: String,
      featureColumnConfigsMap: Map[MergedColumnName, FeatureShardConfiguration],
      numPartitionsOpt: Option[Int]): (DataFrame, Map[MergedColumnName, IndexMapLoader]) =
    readMerged(Seq(path), featureColumnConfigsMap, numPartitionsOpt)

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
   * @param numPartitionsOpt An optional minimum number of partitions for the [[DataFrame]]. If no minimum is provided,
   *                         the default parallelism is used. Spark is generally moving away from manually specifying
   *                         partition counts like this, in favor of inferring it. However, Photon still exposes
   *                         partition counts as a means for tuning job performance. The auto-inferred counts are
   *                         usually much lower than the necessary counts for Photon (especially GAME). This causes a
   *                         lot of shuffling when repartitioning from the auto-partitioned data to the processed GAME
   *                         data. This setting is exposed to allow tuning which can avoid the shuffling.
   * @return The loaded and transformed DataFrame
   */
  def readMerged(
      path: String,
      indexMapLoaders: Map[MergedColumnName, IndexMapLoader],
      featureColumnConfigsMap: Map[MergedColumnName, FeatureShardConfiguration],
      numPartitionsOpt: Option[Int]): DataFrame =
    readMerged(Seq(path), indexMapLoaders, featureColumnConfigsMap, numPartitionsOpt)

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
   * @param numPartitionsOpt An optional minimum number of partitions for the [[DataFrame]]. If no minimum is provided,
   *                         the default parallelism is used. Spark is generally moving away from manually specifying
   *                         partition counts like this, in favor of inferring it. However, Photon still exposes
   *                         partition counts as a means for tuning job performance. The auto-inferred counts are
   *                         usually much lower than the necessary counts for Photon (especially GAME). This causes a
   *                         lot of shuffling when repartitioning from the auto-partitioned data to the processed GAME
   *                         data. This setting is exposed to allow tuning which can avoid the shuffling.
   * @return The loaded and transformed DataFrame
   */
  def readMerged(
      paths: Seq[String],
      featureColumnConfigsMap: Map[MergedColumnName, FeatureShardConfiguration],
      numPartitionsOpt: Option[Int]): (DataFrame, Map[MergedColumnName, IndexMapLoader])

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
   * @param numPartitionsOpt An optional minimum number of partitions for the [[DataFrame]]. If no minimum is provided,
   *                         the default parallelism is used. Spark is generally moving away from manually specifying
   *                         partition counts like this, in favor of inferring it. However, Photon still exposes
   *                         partition counts as a means for tuning job performance. The auto-inferred counts are
   *                         usually much lower than the necessary counts for Photon (especially GAME). This causes a
   *                         lot of shuffling when repartitioning from the auto-partitioned data to the processed GAME
   *                         data. This setting is exposed to allow tuning which can avoid the shuffling.
   * @return The loaded and transformed DataFrame
   */
  def readMerged(
      paths: Seq[String],
      indexMapLoaders: Map[MergedColumnName, IndexMapLoader],
      featureColumnConfigsMap: Map[MergedColumnName, FeatureShardConfiguration],
      numPartitionsOpt: Option[Int]): DataFrame
}
