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

import org.apache.hadoop.fs.{FileSystem, FileUtil, Path}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

/**
 * Some basic functions to deal with NameAndTermFeatureMap: a [[Map]] of feature bag key to [[NameAndTerm]] feature
 * [[RDD]].
 *
 */
object NameAndTermFeatureMapUtils {

  /**
   * Get an index-map as a [[Map]] of feature [[NameAndTerm]] to index for a given set of feature bags.
   *
   * @param nameAndTermFeatureMap The [[Map]] of feature bag key to [[NameAndTerm]] feature [[RDD]]
   * @param featureBagKeys The specified feature bag keys to generate the name to index map explained above
   * @param isAddingIntercept Whether to add a dummy variable to the generated feature map to learn an intercept term
   * @param sc The Spark context
   * @return The generated map from feature name of type [[NameAndTerm]] to feature index of type [[Int]]
   */
  def getFeatureNameAndTermToIndexMap(
      nameAndTermFeatureMap: Map[String, RDD[NameAndTerm]],
      featureBagKeys: Set[String],
      isAddingIntercept: Boolean,
      sc: SparkContext): Map[NameAndTerm, Int] = {

    val featureNameAndTermToIndexMap = nameAndTermFeatureMap
      .filterKeys(featureBagKeys.contains)
      .values
      .fold(sc.emptyRDD[NameAndTerm])(_ ++ _)
      .distinct
      .collect
      .zipWithIndex
      .toMap

    if (isAddingIntercept) {
      featureNameAndTermToIndexMap + (NameAndTerm.INTERCEPT_NAME_AND_TERM -> featureNameAndTermToIndexMap.size)
    } else {
      featureNameAndTermToIndexMap
    }
  }

  /**
   * Save a set of feature names as [[RDD]] of [[NameAndTerm]] to text on HDFS for multiple feature bags.
   *
   * @param nameAndTermFeatureMap The [[Map]] of feature map key to [[NameAndTerm]] feature [[RDD]]
   * @param outputDir The HDFS directory to write the [[NameAndTerm]] feature [[RDD]]s as text files
   * @param sc The Spark context
   */
  def saveAsTextFiles(
      nameAndTermFeatureMap: Map[String, RDD[NameAndTerm]],
      outputDir: String,
      sc: SparkContext): Unit =
    nameAndTermFeatureMap.foreach { case (featureMapKey, featureSet) =>
      val featureSetPath = new Path(outputDir, featureMapKey)
      saveNameAndTermsAsTextFiles(featureSet, featureSetPath, sc)
    }

  /**
   * Save a set of feature names as [[RDD]] of [[NameAndTerm]] to text on HDFS for a single feature bag.
   *
   * @param nameAndTermRDD The [[NameAndTerm]] feature [[RDD]] to be written
   * @param outputPath The HDFS path to which write the [[NameAndTerm]] feature [[RDD]]
   * @param sc The Spark context
   */
  private def saveNameAndTermsAsTextFiles(
      nameAndTermRDD: RDD[NameAndTerm],
      outputPath: Path,
      sc: SparkContext): Unit = {

    val hadoopFS = FileSystem.get(sc.hadoopConfiguration)
    val tmpPath = new Path(outputPath.getParent, s"${outputPath.getName}-tmp")

    // saveAsTextFile in a temporary folder
    hadoopFS.delete(tmpPath, true)
    // save result uncompressed for human-readable output
    sc.hadoopConfiguration.set("mapred.output.compress", "false")
    nameAndTermRDD.map(_.toString).saveAsTextFile(tmpPath.toString)

    // Merge files in temporary folder into final output
    hadoopFS.delete(outputPath, true)
    FileUtil.copyMerge(
      hadoopFS,
      tmpPath,
      hadoopFS,
      outputPath,
      true,
      sc.hadoopConfiguration,
      null)
  }

  /**
   * Parse a set of feature names to [[RDD]] of [[NameAndTerm]] from text for multiple feature bags.
   *
   * @param inputDir The input HDFS directory
   * @param featureBagKeys The set of feature bag keys to look for from the input directory
   * @param sc The Spark context
   * @return A [[Map]] of feature bag key to [[NameAndTerm]] feature [[RDD]] parsed from text files on HDFS
   */
  protected[ml] def readNameAndTermFeatureMapFromTextFiles(
      inputDir: Path,
      featureBagKeys: Set[String],
      sc: SparkContext): Map[String, RDD[NameAndTerm]] =
    featureBagKeys
      .map { featureBagKey =>
        val inputPath = new Path(inputDir, featureBagKey)
        val nameAndTermFeatureSet = readNameAndTermRDDFromTextFiles(inputPath, sc)

        (featureBagKey, nameAndTermFeatureSet)
      }
      .toMap

  /**
   * Parse a set of feature names to [[RDD]] of [[NameAndTerm]] from text for a single feature bag.
   *
   * @param inputPath The input path
   * @param sc The Spark context
   * @return The [[RDD]] of [[NameAndTerm]] read from the text files of the given input path
   */
  private def readNameAndTermRDDFromTextFiles(inputPath: Path, sc: SparkContext): RDD[NameAndTerm] =
    sc.textFile(inputPath.toString)
      .map { string =>
        string.split(NameAndTerm.STRING_DELIMITER) match {
          case Array(name, term) =>
            NameAndTerm(name, term)

          case Array(name) =>
            NameAndTerm(name, "")

          case other =>
            throw new UnsupportedOperationException(
              s"Unexpected entry $string when parsing it to NameAndTerm, the expected number of tokens is 1 or 2, " +
                s"but found ${other.length}}.")
        }
      }
}
