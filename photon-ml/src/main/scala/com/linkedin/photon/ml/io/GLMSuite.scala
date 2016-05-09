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
package com.linkedin.photon.ml.io

import java.io.IOException
import java.lang.{Double => JDouble}
import java.util.{List => JList, Map => JMap}

import breeze.linalg.SparseVector
import FieldNamesType._
import com.linkedin.photon.avro.generated.FeatureSummarizationResultAvro
import com.linkedin.photon.ml.avro.{TrainingExampleFieldNames, ResponsePredictionFieldNames, AvroIOUtils}
import com.linkedin.photon.ml.data
import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.stat.BasicStatisticalSummary
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import com.linkedin.photon.ml.util._
import org.apache.avro.generic.GenericRecord
import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import scala.collection.JavaConversions.mapAsJavaMap
import scala.collection.mutable
import scala.util.parsing.json.JSON


/**
 * A suite responsible for transforming raw data into [[data.LabeledPoint]]
 * and write the learned [[GeneralizedLinearModel]] in text or avro files.
  *
  * @param fieldNamesType Input Avro file's format, which contains the information of each field's name
 * @param addIntercept Whether to add the an additional variable "1" to the feature vector for intercept learning
 *                     purpose
 * @author xazhang
 * @author nkatariy
 */
// TODO we already have Params class handling all parameters, passing in arguments here seems to be a very trivial
// and fragile style that could easily require method signature replacements
@SerialVersionUID(2L) // NOTE: Remember to change this if you add new member fields / make significant API modifications
class GLMSuite(
    fieldNamesType: FieldNamesType,
    addIntercept: Boolean,
    constraintString: Option[String],
    offHeapIndexMapLoader: Option[IndexMapLoader] = None)
  extends Serializable {

  /**
   * The input avro files' field names
   */
  private val fieldNames = fieldNamesType match {
    case RESPONSE_PREDICTION => ResponsePredictionFieldNames
    case TRAINING_EXAMPLE => TrainingExampleFieldNames
    case _ => throw new IllegalArgumentException(s"Input training file's field name type cannot be $fieldNamesType")
  }

  /**
   * Mapping the String based feature names to integer based Ids, for more efficient memory usage when persisting data
   * into memory. Making it transient in order to avoid it being serialized to the executors, which could be expensive
   * and is unnecessary.
   */
  @transient var featureKeyToIdMap: IndexMap = null

  /* Map of feature indices to their (lowerBound, upperBound) constraints */
  @transient var constraintFeatureMap: Option[Map[Int, (Double, Double)]] = None

  /* set of selected features. If empty, all features are used */
  @transient var selectedFeatures: Set[String] = Set.empty[String]

  private var _indexMapLoader: IndexMapLoader = null

  /**
   * Read the [[data.LabeledPoint]] from a directory of Avro files
    *
    * @param sc The Spark context
   * @param inputDir Input directory of the Avro files
   * @param selectedFeaturesFile Path to the file containing features that should be used in training. This file is
   *                             expected to be an avro file containing records with the schema FeatureNameTermAvro
   * @param minNumPartitions Set the minimum number of Hadoop splits to generate. This would be potentially helpful when
   *                         the number of default Hadoop splits is small. Note that when the default number of Hadoop
   *                         splits (from HDFS) is larger than minNumPartitions, then minNumPartitions will be ignored
   *                         and the number of partitions of the resulting RDD will be same as the default number of
   *                         Hadoop splits. In short, minNumPartitions will *only* be able to increase the number of
   *                         partitions.
   * @return Tuple of (number of features in dataset, number of selected features, A RDD of [[data.LabeledPoint]])
   */
  def readLabeledPointsFromAvro(sc: SparkContext, inputDir: String,
                                selectedFeaturesFile: Option[String], minNumPartitions: Int): RDD[LabeledPoint] = {
    val avroRDD = AvroIOUtils.readFromAvro[GenericRecord](sc, inputDir, minNumPartitions)
    if (selectedFeatures.isEmpty) {
      selectedFeatures = getSelectedFeatureSetFromFile(sc, selectedFeaturesFile)
    }
    /*Only load the featureKeyToIdMap once*/
    if (featureKeyToIdMap == null) {
      _indexMapLoader = offHeapIndexMapLoader match {
        case Some(loader) => loader
        case None =>
          // Build the default indexmap if offheap map dir is not provided
          _indexMapLoader = createDefaultIndexMapLoader(avroRDD, selectedFeatures)
          // Important to call prepare, though params are not actually needed for DefaultIndexMapLoader.
          // TODO More refactoring is needed here that indexMap creation logic should be put outside of this
          // suite class. It tries to do too many things at once.
          _indexMapLoader.prepare(sc, null)
          _indexMapLoader
      }

      featureKeyToIdMap = _indexMapLoader.indexMapForDriver()
    }
    if (constraintFeatureMap.isEmpty) {
      constraintFeatureMap = createConstraintFeatureMap()
    }
    toLabeledPoints(avroRDD)
  }

  /**
   * Takes the selected features file and returns the set of keys corresponding to the these features
   *
   * @param sc The spark context
   * @param selectedFeaturesFile Path to the file containing features that should be used in training. This file is
   *                             expected to be an avro file containing records with the schema FeatureNameTermAvro
   * @return set of selected features
   */
  def getSelectedFeatureSetFromFile(sc: SparkContext, selectedFeaturesFile: Option[String]): Set[String] = {
    selectedFeaturesFile match {
      case Some(filename: String) =>  AvroIOUtils.readFromAvro[GenericRecord](sc, filename, 1)
        .map(x => Utils.getFeatureKey(x, "name", "term", GLMSuite.DELIMITER))
        .collect().toSet
      case _ => Set.empty[String]
    }
  }

  /**
   * Load the featureKeyToIdMap that maps the String based feature keys into Integer based feature Ids. Only
   * use features provided by the given selected features set. If the set is empty however, all features will
   * be used.
   *
   * @param avroRDD The avro files that contains feature information
   * @param selectedFeatures Set of selected features in the feature key form ({@see Utils#getFeatureKey})
   * @return Tuple of (number of distinct features, map that maps String based features keys to integer based
   *   featureIds)
   */
  private def createDefaultIndexMapLoader[T <: GenericRecord](avroRDD: RDD[T],
                                                        selectedFeatures: Set[String]): IndexMapLoader = {
    def getFeatures(avroRecord: GenericRecord): Array[String] = {
      avroRecord.get(fieldNames.features) match {
        case recordList: JList[_] =>
          recordList.toArray.map { case record: GenericRecord =>
            Utils.getFeatureKey(record, fieldNames.name, fieldNames.term, GLMSuite.DELIMITER)
          }
        case other =>
          throw new IOException(s"Avro field [${fieldNames.features}] (val = ${other.toString}) is not a list")
      }
    }

    val featureRDD = avroRDD.flatMap { k => getFeatures(k) }.distinct()
    val featureSet = if (selectedFeatures.isEmpty) {
      featureRDD.collect().toSet
    } else {
      featureRDD.filter(selectedFeatures.contains).collect().toSet
    }

    if (addIntercept) {
      new DefaultIndexMapLoader((featureSet + GLMSuite.INTERCEPT_NAME_TERM).zipWithIndex.toMap)
    } else {
      new DefaultIndexMapLoader(featureSet.zipWithIndex.toMap)
    }
  }

  /**
   * Take the constraint string which is a JSON array of maps to create the constraint map from feature index to their
   * bounds that can be used by the optimizers. There are several expectations from the input constraint string which
   * if violated, an exception will be thrown
   * 1. Every map in the constraint string is expected to contain both [[ConstraintMapKeys.name]] and
   *    [[ConstraintMapKeys.term]] keys
   * 2. The lower bound must not be greater than the upper bound in some constraint
   * 3. If the name is a wildcard, the term must also be a wildcard. Currently, we only support wildcards in term or in
   *    both which implies the constraint is to be applied to all features
   * 4. There must not be an overlap among constraints. For instance, specifying an explicit feature constraint as well
   *    as a wildcard constraint that is applicable to that same feature or specifying an all-feature constraint using
   *    a wildcard in both name and term as well as specifying some individual feature constraints are examples of
   *    overlaps. Please note that we flag the moment we see the same feature and we do not check whether the earlier
   *    constraint is same as the specified overlap
   *
   * @return None if the map is empty at the end else return the constraint map
   */
  @throws(classOf[IllegalArgumentException])
  def createConstraintFeatureMap(): Option[Map[Int, (Double, Double)]] = {
    val constraintMap = mutable.Map[Int, (Double, Double)]()
    constraintString match {
      case Some(x) =>
        val parsedConstraints = JSON.parseFull(x)
        parsedConstraints match {
          case Some(parsed: List[Map[String, Any]]) =>
            parsed.foreach(entry => {
              val message = s"Each map in the constraint map is expected to have the feature name field specified. " +
                  s"The input constraint string was [$constraintString] and the malformed map was [$entry]"
              val name = Utils.getKeyFromMapOrElse[String](entry, ConstraintMapKeys.name.toString, Left(message))
              val term = Utils.getKeyFromMapOrElse[String](entry, ConstraintMapKeys.term.toString, Left(message))
              val lowerBound = Utils.getKeyFromMapOrElse[Double](entry, ConstraintMapKeys.lowerBound.toString,
                Right(Double.NegativeInfinity))
              val upperBound = Utils.getKeyFromMapOrElse[Double](entry, ConstraintMapKeys.upperBound.toString,
                Right(Double.PositiveInfinity))

              require(lowerBound > Double.NegativeInfinity || upperBound < Double.PositiveInfinity,
                s"The lower and upper bound are respectively -Inf and +Inf for the feature with name [$name] and " +
                s"term [$term]. This is an invalid constraint specification.")

              require(lowerBound < upperBound,
                s"The lower bound [$lowerBound] is incorrectly specified as greater than the upper bound " +
                s"[$upperBound] for the feature with name [$name] and term [$term].")

              if (name == GLMSuite.WILDCARD) {
                if (term == GLMSuite.WILDCARD) {
                  if (constraintMap.nonEmpty) {
                    throw new IllegalArgumentException(s"Potentially conflicting constraints specified. When the " +
                      s"name and term are specified as wildcards, it is expected that no other constraints are" +
                      s" specified. The specified constraint string was [$constraintString]")
                  } else {
                    featureKeyToIdMap.foreach(x =>
                      if (!x._1.equals(GLMSuite.INTERCEPT_NAME_TERM)) {
                        constraintMap.put(x._2, (lowerBound, upperBound))
                      })
                  }
                } else {
                  throw new IllegalArgumentException("We do not support wildcard in feature name alone for now. If " +
                    "the name is a wildcard, it is expected that the term is also a wildcard. Wildcards in names " +
                    "but not in term may potentially be incorporated later as feature requests")
                }
              } else if (term == GLMSuite.WILDCARD) {
                featureKeyToIdMap
                  .filter(x => x._1.startsWith(name + GLMSuite.DELIMITER))
                  .foreach(x => {
                    if (constraintMap.containsKey(x._2)) {
                      throw new IllegalArgumentException(s"Please avoid specifying potentially " +
                        s"conflicting bounds. The feature with name [$name] and term " +
                        s"[${Utils.getFeatureTermFromKey(x._1)}] was already added with bounds " +
                        s"[${constraintMap.get(x._2)}] and attempted to add it back with bounds " +
                        s"[${(lowerBound, upperBound)}]")
                    } else {
                      constraintMap.put(x._2, (lowerBound, upperBound))
                    }
                  })
              } else {
                featureKeyToIdMap.get(Utils.getFeatureKey(name, term))
                  .foreach(x => {
                    if (constraintMap.containsKey(x)) {
                      throw new IllegalArgumentException(s"Please avoid specifying potentially " +
                          s"conflicting bounds. The feature with name [$name] and term [$term] was " +
                          s"already added with bounds [${constraintMap.get(x)}] and attempted to add " +
                          s"it back with bounds [${(lowerBound, upperBound)}]")
                    } else {
                      constraintMap.put(x, (lowerBound, upperBound))
                    }
                  })
              }
            })
        }
        if (constraintMap.nonEmpty) {
          Some(Map[Int, (Double, Double)]() ++ constraintMap)
        } else {
          None
        }
      case _ => None
    }
  }

  /**
   * Transform the Avro files into LabeledPoints
    *
    * @param avroRDD A RDD of Avro files
   * @return A RDD of [[data.LabeledPoint]]
   */
  private def toLabeledPoints[T <: GenericRecord](avroRDD: RDD[T]): RDD[LabeledPoint] = {
    // Returns None if no features in the record are in the featureKeyToIdMap
    def parseAvroRecord(avroRecord: GenericRecord, indexMap: IndexMap): Option[LabeledPoint] = {
      val numFeatures = indexMap.size

      val features = avroRecord.get(fieldNames.features) match {
        case recordList: JList[_] =>
          val nnz =
            if (addIntercept) {
              recordList.size() + 1
            } else {
              recordList.size()
            }
          val pairsArr = new mutable.ArrayBuffer[(Int, Double)](nnz)
          val iter = recordList.iterator
          while (iter.hasNext) {
            iter.next match {
              case record: GenericRecord =>
                val featureFullName = Utils.getFeatureKey(record, fieldNames.name, fieldNames.term, GLMSuite.DELIMITER)
                val idx = indexMap.getIndex(featureFullName)
                if (idx != IndexMap.NULL_KEY) {
                  pairsArr += ((idx, Utils.getDoubleAvro(record, fieldNames.value)))
                }
              case any =>
                throw new IOException(s"${String.valueOf(any)} in ${fieldNames.features} list is not a record")
            }
          }
          if (addIntercept) {
            val featureFullName = GLMSuite.INTERCEPT_NAME_TERM
            pairsArr += ((indexMap.getIndex(featureFullName), 1.0))
          }
          val sortedPairsArray = pairsArr.toArray.sortBy(_._1)
          val index = sortedPairsArray.map(_._1)
          val value = sortedPairsArray.map(_._2)
          new SparseVector[Double](index, value, numFeatures)
        case other =>
          throw new IOException(s"Avro field [${fieldNames.features}] (val = ${String.valueOf(other)}) is not a list")
      }
      if (features.activeSize == 0 || (addIntercept && features.activeSize == 1)) {
        None
      } else {
        val response = Utils.getDoubleAvro(avroRecord, fieldNames.response)
        val offset =
          if (avroRecord.get(fieldNames.offset) != null) {
            Utils.getDoubleAvro(avroRecord, fieldNames.offset)
          } else {
            0
          }
        val weight =
          if (avroRecord.get(fieldNames.weight) != null) {
            Utils.getDoubleAvro(avroRecord, fieldNames.weight)
          } else {
            1
          }
        Some(new LabeledPoint(response, features, offset, weight))
      }
    }

    avroRDD.mapPartitions { iter =>
      val map = _indexMapLoader.indexMapForRDD()
      iter.flatMap { r => parseAvroRecord(r, map) }
    }
  }

  /**
   * Write a map of learned [[GeneralizedLinearModel]] to text files
    *
    * @param sc The Spark context
   * @param models The map of (Model Id -> [[GeneralizedLinearModel]])
   * @param modelDir The directory for the output text files
   */
  def writeModelsInText(
                           sc: SparkContext,
                           models: Iterable[(Double, GeneralizedLinearModel)],
                           modelDir: String) {


    println("Models SIZE: " + models.size)
    models.foreach{ case (lambda, m) => println(lambda + ": " + m.toString)}

    sc.parallelize(models.toSeq, models.size).mapPartitions { iter =>
        val indexMap = _indexMapLoader.indexMapForRDD()

        val modelStrs = new mutable.ArrayBuffer[String]()

        while (iter.hasNext) {
          val t = iter.next()
          val regWeight = t._1
          val model = t._2

          val builder = new mutable.ArrayBuffer[String]()
          model.intercept match {
            case Some(intercept) =>
              val tokens = GLMSuite.INTERCEPT_NAME_TERM.split(GLMSuite.DELIMITER)
              if (tokens.length == 1) {
                builder += s"${tokens(0)}\t${""}\t$intercept\t$regWeight"
              } else if (tokens.length == 2) {
                builder += s"${tokens(0)}\t${tokens(1)}\t$intercept\t$regWeight"
              } else {
                throw new IOException(s"unexpected intercept name: ${GLMSuite.INTERCEPT_NAME_TERM}")
              }
            case None =>
          }
          model.coefficients.toArray.zipWithIndex.sortWith((p1, p2) => p1._1 > p2._1).foreach { case (value, index) =>
            val nameAndTerm = indexMap.getFeatureName(index)
            nameAndTerm.foreach { s =>
                val tokens = s.split(GLMSuite.DELIMITER)
                if (tokens.length == 1) {
                  builder += s"${tokens(0)}\t${""}\t$value\t$regWeight"
                } else if (tokens.length == 2) {
                  builder += s"${tokens(0)}\t${tokens(1)}\t$value\t$regWeight"
                } else {
                  throw new IOException(s"unknown name and terms: $s")
                }
            }
          }
          val s = builder.mkString("\n")
          modelStrs += s
      }

      modelStrs.iterator
    }.saveAsTextFile(modelDir)
  }

  /**
   * Write basic feature statistics in Avro format
   *
   * @param sc Spark context
   * @param summary The summary of the features
   * @param outputDir Output directory
   */
  def writeBasicStatistics(sc: SparkContext, summary: BasicStatisticalSummary, outputDir: String): Unit = {
    val keyToIdMap = featureKeyToIdMap
    def featureStringToTuple(str: String): (String, String) = {
      val splits = str.split(GLMSuite.DELIMITER)
      if (splits.length == 2) {
        (splits(0), splits(1))
      } else {
        (splits(0), "")
      }
    }
    val featureTuples = keyToIdMap.toArray
      .sortBy[Int] { case (key, id) => id }
      .map { case (key, id) => featureStringToTuple(key) }

    val summaryList = List(
      summary.max.toArray, summary.min.toArray, summary.mean.toArray, summary.normL1.toArray, summary.normL2.toArray,
      summary.numNonzeros.toArray, summary.variance.toArray)
        .transpose
        .map {
          case List(max, min, mean, normL1, normL2, numNonZeros, variance) =>
            new BasicSummaryItems(max, min, mean, normL1, normL2, numNonZeros, variance)
        }

    val outputAvro = featureTuples.zip(summaryList).map {
      case ((name, term), items) =>
        val jMap: JMap[CharSequence, JDouble] = mapAsJavaMap(
          Map(
            "max" -> items.max,
            "min" -> items.min,
            "mean" -> items.mean,
            "normL1" -> items.normL1,
            "normL2" -> items.normL2,
            "numNonzeros" -> items.numNonzeros,
            "variance" -> items.variance))

        FeatureSummarizationResultAvro.newBuilder()
          .setFeatureName(name)
          .setFeatureTerm(term)
          .setMetrics(jMap).build()
    }
    val outputFile = new Path(outputDir, GLMSuite.DEFAULT_AVRO_FILE_NAME).toString
    AvroIOUtils.saveAsSingleAvro(
      sc, outputAvro, outputFile, FeatureSummarizationResultAvro.getClassSchema.toString, forceOverwrite = true)
  }

  /**
   * Get the intercept index. This is used especially for normalization because the intercept should be treated
   * differently.
    *
    * @return The option for the intercept index value
   */
  def getInterceptId: Option[Int] = featureKeyToIdMap.get(GLMSuite.INTERCEPT_NAME_TERM)
}

private case class BasicSummaryItems(
  max: Double, min: Double, mean: Double, normL1: Double, normL2: Double, numNonzeros: Double, variance: Double)

protected[ml] object GLMSuite {
  /**
   * Delimiter used to concatenate feature name and term into feature key
   */
  val DELIMITER = "\u0001"

  /**
   * Wildcard character used for specifying the feature constraints. Only the term is allowed to be a wildcard normally
   * unless one wants to apply bounds to all features in which case both name and term can be specified as wildcards.
   * Currently, we do not support wildcards in name alone.
   */
  val WILDCARD = "*"

  /**
   * Name of the intercept
   */
  val INTERCEPT_NAME = "(INTERCEPT)"
  val INTERCEPT_TERM = ""
  val INTERCEPT_NAME_TERM = Utils.getFeatureKey(INTERCEPT_NAME, INTERCEPT_TERM)
  val DEFAULT_AVRO_FILE_NAME = "part-00000.avro"
}
