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
package com.linkedin.photon.ml.io.deprecated

import java.io.IOException
import java.util.{List => JList}

import scala.collection.JavaConversions.mapAsJavaMap
import scala.collection.mutable
import scala.util.parsing.json.JSON

import breeze.linalg.SparseVector
import org.apache.avro.generic.GenericRecord
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.data.avro.{AvroUtils, ResponsePredictionFieldNames, TrainingExampleFieldNames}
import com.linkedin.photon.ml.io.deprecated.FieldNamesType._
import com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel
import com.linkedin.photon.ml.util._
import com.linkedin.photon.ml.{Constants, data}

/**
 * A suite responsible for transforming raw data into [[data.LabeledPoint]]
 * and write the learned [[GeneralizedLinearModel]] in text or avro files.
 *
 * @param fieldNamesType Input Avro file's format, which contains the information of each field's name
 * @param addIntercept Whether to add the an additional variable "1" to the feature vector for intercept learning
 *                     purpose
 */
@SerialVersionUID(2L) // NOTE Remember to change this if you add new member fields / make significant API modifications
class GLMSuite(
    fieldNamesType: FieldNamesType,
    addIntercept: Boolean,
    constraintString: Option[String],
    offHeapIndexMapLoader: Option[IndexMapLoader] = None)
  extends Serializable {

  // TODO we already have Params class handling all parameters, passing in arguments here seems to be a very trivial
  // and fragile style that could easily require method signature replacements

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
  @transient var featureKeyToIdMap: IndexMap = null // null intentional

  /* Map of feature indices to their (lowerBound, upperBound) constraints */
  @transient var constraintFeatureMap: Option[Map[Int, (Double, Double)]] = None

  /* set of selected features. If empty, all features are used */
  @transient var selectedFeatures: Set[String] = Set.empty[String]

  private var _indexMapLoader: IndexMapLoader = null // null intentional

  /**
   *
   * @return
   */
  def indexMapLoader(): IndexMapLoader = _indexMapLoader

  /**
   * Read the [[data.LabeledPoint]] from a directory of Avro files.
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
  def readLabeledPointsFromAvro(
      sc: SparkContext,
      inputDir: String,
      selectedFeaturesFile: Option[String],
      minNumPartitions: Int): RDD[LabeledPoint] = {

    val avroRDD = AvroUtils.readAvroFilesInDir[GenericRecord](sc, inputDir, minNumPartitions)

    if (selectedFeatures.isEmpty) {
      selectedFeatures = getSelectedFeatureSetFromFile(sc, selectedFeaturesFile)
    }

    // Only load the featureKeyToIdMap once
    if (featureKeyToIdMap == null) {
      // Build the default indexmap if offheap map dir is not provided
      // TODO More refactoring is needed here that indexMap creation logic should be put outside of this
      // suite class. It tries to do too many things at once.
      _indexMapLoader = offHeapIndexMapLoader match {
        case Some(loader) => loader
        case None => createDefaultIndexMapLoader(sc, avroRDD, selectedFeatures)
      }
      featureKeyToIdMap = _indexMapLoader.indexMapForDriver()
    }

    if (constraintFeatureMap.isEmpty) {
      constraintFeatureMap = createConstraintFeatureMap()
    }

    toLabeledPoints(avroRDD)
  }

  /**
   * Takes the selected features file and returns the set of keys corresponding to the these features.
   *
   * @param sc The spark context
   * @param selectedFeaturesFile Path to the file containing features that should be used in training. This file is
   *                             expected to be an avro file containing records with the schema FeatureNameTermAvro
   * @return Set of selected features
   */
  def getSelectedFeatureSetFromFile(
      sc: SparkContext,
      selectedFeaturesFile: Option[String]): Set[String] = selectedFeaturesFile match {
    case Some(filename: String) =>  AvroUtils.readAvroFilesInDir[GenericRecord](sc, filename, 1)
      .map(x => Utils.getFeatureKey(x, "name", "term", Constants.DELIMITER))
      .collect().toSet
    case _ => Set.empty[String]
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
  private def createDefaultIndexMapLoader[T <: GenericRecord](
      sc: SparkContext,
      avroRDD: RDD[T],
      selectedFeatures: Set[String]): IndexMapLoader = {

    def getFeatures(avroRecord: GenericRecord): Array[String] = {
      avroRecord.get(fieldNames.FEATURES) match {
        case recordList: JList[_] =>
          recordList.toArray.map { case record: GenericRecord =>
            Utils.getFeatureKey(record, fieldNames.NAME, fieldNames.TERM, Constants.DELIMITER)
          }
        case other =>
          throw new IOException(s"Avro field [${fieldNames.FEATURES}] (val = ${other.toString}) is not a list")
      }
    }

    val featureRDD = avroRDD.flatMap { k => getFeatures(k) }.distinct()
    val featureSet = if (selectedFeatures.isEmpty) {
      featureRDD.collect().toSet
    } else {
      featureRDD.filter(selectedFeatures.contains).collect().toSet
    }

    if (addIntercept) {
      new DefaultIndexMapLoader(sc, (featureSet + Constants.INTERCEPT_KEY).zipWithIndex.toMap)
    } else {
      new DefaultIndexMapLoader(sc, featureSet.zipWithIndex.toMap)
    }
  }

  /**
   * Take the constraint string which is a JSON array of maps to create the constraint map from feature index to their
   * bounds that can be used by the optimizers. There are several expectations from the input constraint string which
   * if violated, an exception will be thrown:
   * 1. Every map in the constraint string is expected to contain both [[ConstraintMapKeys.name]] and
   *    [[ConstraintMapKeys.term]] keys.
   * 2. The lower bound must not be greater than the upper bound in some constraint.
   * 3. If the name is a wildcard, the term must also be a wildcard. Currently, we only support wildcards in term or in
   *    both which implies the constraint is to be applied to all features.
   * 4. There must not be an overlap among constraints. For instance, specifying an explicit feature constraint as well
   *    as a wildcard constraint that is applicable to that same feature or specifying an all-feature constraint using
   *    a wildcard in both name and term as well as specifying some individual feature constraints are examples of
   *    overlaps. Please note that we flag the moment we see the same feature and we do not check whether the earlier
   *    constraint is same as the specified overlap.
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
          case Some(p: Any) =>
            val parsed = p.asInstanceOf[List[Map[String, Any]]] // to avoid warning about type erasure
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

              if (name == Constants.WILDCARD) {
                if (term == Constants.WILDCARD) {
                  if (constraintMap.nonEmpty) {
                    throw new IllegalArgumentException(s"Potentially conflicting constraints specified. When the " +
                      s"name and term are specified as wildcards, it is expected that no other constraints are" +
                      s" specified. The specified constraint string was [$constraintString]")
                  } else {
                    featureKeyToIdMap.foreach(x =>
                      if (!x._1.equals(Constants.INTERCEPT_KEY)) {
                        constraintMap.put(x._2, (lowerBound, upperBound))
                      })
                  }
                } else {
                  throw new IllegalArgumentException("We do not support wildcard in feature name alone for now. If " +
                    "the name is a wildcard, it is expected that the term is also a wildcard. Wildcards in names " +
                    "but not in term may potentially be incorporated later as feature requests")
                }
              } else if (term == Constants.WILDCARD) {
                featureKeyToIdMap
                  .filter(x => x._1.startsWith(name + Constants.DELIMITER))
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
          case _ => throw new RuntimeException("Shouldn't be here")
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
   * Transform the Avro files into LabeledPoints.
   *
   * @param avroRDD A RDD of Avro files
   * @return A RDD of [[data.LabeledPoint]]
   */
  private def toLabeledPoints[T <: GenericRecord](avroRDD: RDD[T]): RDD[LabeledPoint] = {
    // Returns None if no features in the record are in the featureKeyToIdMap
    def parseAvroRecord(avroRecord: GenericRecord, indexMap: IndexMap): LabeledPoint = {
      val numFeatures = indexMap.size

      val features = avroRecord.get(fieldNames.FEATURES) match {
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
                val featureFullName = Utils.getFeatureKey(record, fieldNames.NAME, fieldNames.TERM, Constants.DELIMITER)
                val idx = indexMap.getIndex(featureFullName)
                if (idx != IndexMap.NULL_KEY) {
                  pairsArr += ((idx, Utils.getDoubleAvro(record, fieldNames.VALUE)))
                }
              case any =>
                throw new IOException(s"${String.valueOf(any)} in ${fieldNames.FEATURES} list is not a record")
            }
          }
          if (addIntercept) {
            val featureFullName = Constants.INTERCEPT_KEY
            pairsArr += ((indexMap.getIndex(featureFullName), 1.0))
          }
          val sortedPairsArray = pairsArr.toArray.sortBy(_._1)
          val index = sortedPairsArray.map(_._1)
          val value = sortedPairsArray.map(_._2)
          new SparseVector[Double](index, value, numFeatures)
        case other =>
          throw new IOException(s"Avro field [${fieldNames.FEATURES}] (val = ${String.valueOf(other)}) is not a list")
      }

      val response = Utils.getDoubleAvro(avroRecord, fieldNames.RESPONSE)
      val offset =
        if (avroRecord.get(fieldNames.OFFSET) != null) {
          Utils.getDoubleAvro(avroRecord, fieldNames.OFFSET)
        } else {
          0
        }
      val weight =
        if (avroRecord.get(fieldNames.WEIGHT) != null) {
          Utils.getDoubleAvro(avroRecord, fieldNames.WEIGHT)
        } else {
          1
        }

      require(weight >= 0, "Found sample with negative weight.")

      new LabeledPoint(response, features, offset, weight)
    }

    val labeledPoints = avroRDD
      .mapPartitions { iter =>
        val map = _indexMapLoader.indexMapForRDD()
        iter.map { r => parseAvroRecord(r, map) }
      }
      .filter(aPoint => aPoint.weight > 0.0)

    // TODO: This is caught later (Photon Driver.prepareTrainingData) in a check that we have at least 1 training sample
    //require(labeledPoints.count > 0)

    labeledPoints
  }

  /**
   * Get the intercept index. This is used especially for normalization because the intercept should be treated
   * differently.
   *
   * @return The option for the intercept index value
   */
  def getInterceptId: Option[Int] = featureKeyToIdMap.get(Constants.INTERCEPT_KEY)
}
