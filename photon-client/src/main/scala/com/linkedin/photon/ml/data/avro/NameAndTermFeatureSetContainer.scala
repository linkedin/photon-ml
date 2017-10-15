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
package com.linkedin.photon.ml.data.avro

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext

import com.linkedin.photon.ml.util.IOUtils

/**
 * A wrapper class for a map of feature section key to a set of [[NameAndTerm]] features
 *
 * @param nameAndTermFeatureSets A [[Map]] of feature section key to [[NameAndTerm]] feature sets
 */
protected[ml] class NameAndTermFeatureSetContainer(nameAndTermFeatureSets: Map[String, Set[NameAndTerm]]) {

  /**
   * Get the map from feature name of type [[NameAndTerm]] to feature index of type [[Int]] based on the specified
   * feature section keys from the input.
   *
   * @param featureSectionKeys The specified feature section keys to generate the name to index map explained above
   * @param isAddingIntercept Whether to add a dummy variable to the generated feature map to learn an intercept term
   * @return The generated map from feature name of type [[NameAndTerm]] to feature index of type [[Int]]
   */
  def getFeatureNameAndTermToIndexMap(
      featureSectionKeys: Set[String],
      isAddingIntercept: Boolean): Map[NameAndTerm, Int] = {

    val featureNameAndTermToIndexMap = nameAndTermFeatureSets
      .filterKeys(featureSectionKeys.contains)
      .values
      .fold(Set[NameAndTerm]())(_ ++ _)
      .zipWithIndex
      .toMap

    if (isAddingIntercept) {
      featureNameAndTermToIndexMap + (NameAndTerm.INTERCEPT_NAME_AND_TERM -> featureNameAndTermToIndexMap.size)
    } else {
      featureNameAndTermToIndexMap
    }
  }

  /**
   * Write each of the feature map to HDFS.
   *
   * @param nameAndTermFeatureSetContainerOutputDir The HDFS directory to write the feature sets as text files
   * @param sc The Spark context
   */
  def saveAsTextFiles(nameAndTermFeatureSetContainerOutputDir: String, sc: SparkContext): Unit = {
    nameAndTermFeatureSets.foreach { case (featureSectionKey, featureSet) =>
      val featureSetPath = new Path(nameAndTermFeatureSetContainerOutputDir, featureSectionKey)
      saveNameAndTermSetAsTextFiles(featureSet, sc, featureSetPath)
    }
  }

  /**
   * Write the [[Set]] of [[NameAndTerm]]s to HDFS as text files.
   *
   * @param nameAndTermSet The map to be written
   * @param sc The Spark context
   * @param outputPath The HDFS path to which write the map
   */
  private def saveNameAndTermSetAsTextFiles(
      nameAndTermSet: Set[NameAndTerm],
      sc: SparkContext,
      outputPath: Path): Unit = {

    val iterator = nameAndTermSet.iterator.map(_.toString)
    IOUtils.writeStringsToHDFS(iterator, outputPath, sc.hadoopConfiguration, forceOverwrite = false)
  }
}

object NameAndTermFeatureSetContainer {

  /**
   * Parse the [[NameAndTermFeatureSetContainer]] from text files on HDFS.
   *
   * @param nameAndTermFeatureSetContainerInputDir The input HDFS directory
   * @param featureSectionKeys The set of feature section keys to look for from the input directory
   * @param configuration The Hadoop configuration
   * @return This [[NameAndTermFeatureSetContainer]] parsed from text files on HDFS
   */
  protected[ml] def readNameAndTermFeatureSetContainerFromTextFiles(
      nameAndTermFeatureSetContainerInputDir: Path,
      featureSectionKeys: Set[String],
      configuration: Configuration): NameAndTermFeatureSetContainer = {

    val nameAndTermFeatureSets = featureSectionKeys
      .map { featureSectionKey =>
        val inputPath = new Path(nameAndTermFeatureSetContainerInputDir, featureSectionKey)
        val nameAndTermFeatureSet = readNameAndTermSetFromTextFiles(inputPath, configuration)
        (featureSectionKey, nameAndTermFeatureSet)
      }
      .toMap

    new NameAndTermFeatureSetContainer(nameAndTermFeatureSets)
  }

  /**
   * Read a [[Set]] of [[NameAndTerm]] from the text files within the input path.
   *
   * @param inputPath The input path
   * @param configuration the Hadoop configuration
   * @return The [[Set]] of [[NameAndTerm]] read from the text files of the given input path
   */
  private def readNameAndTermSetFromTextFiles(inputPath: Path, configuration: Configuration): Set[NameAndTerm] =
    IOUtils
      .readStringsFromHDFS(inputPath, configuration)
      .map { string =>
        string.split("\t") match {
          case Array(name, term) => NameAndTerm(name, term)
          case Array(name) => NameAndTerm(name, "")
          case other => throw new UnsupportedOperationException(
            s"Unexpected entry $string when parsing it to NameAndTerm, " +
              s"after splitting by tab the expected number of tokens is 1 or 2, but found ${other.length}}.")
        }
      }
      .toSet
}
