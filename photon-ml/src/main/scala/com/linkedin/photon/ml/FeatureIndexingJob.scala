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
package com.linkedin.photon.ml


import com.linkedin.photon.ml.avro.model.TrainingExampleFieldNames
import com.linkedin.photon.ml.io.FieldNamesType._

import scala.collection.mutable

import com.linkedin.photon.ml.avro.{ResponsePredictionFieldNames, FieldNames, AvroIOUtils}
import com.linkedin.photon.ml.io.{FieldNamesType, GLMSuite}
import com.linkedin.photon.ml.util.{Utils, PalDBIndexMapBuilder}
import org.apache.avro.generic.GenericRecord
import org.apache.spark.rdd.RDD
import org.apache.spark.{HashPartitioner, SparkConf, SparkContext}


import java.util.{List => JList}

/**
  * A class that builds feature index map as an independent Spark job. Recommended when feature space is large,
  * typically when there are more than 200k unique features.
  *
  * The job expected three required arguments and one optional one:
  * [input_data_path]: The input path of data
  * [partition_num]: The number of partitions to break the storage into. This is merely introduced as an optimization at
  *   build stage that we don't necessary have to shuffle all features into the same partition but rather we could
  *   just hold them into multiple ones. A heuristically good number could be 1-10, or even as large as 20; but please
  *   avoid setting it into an arbitrarily large number. Use just 1 or 2 if you find the indexing job is already fast
  *   enough.
  * [output_dir]: The output directory
  * [if_add_intercept] (optional, default=true): whether include INTERCEPT into the map.
  *
  */
class FeatureIndexingJob(val sc: SparkContext,
                         val inputPath: String,
                         val partitionNum: Int,
                         val outputPath: String,
                         val addIntercept: Boolean,
                         val fieldNames: FieldNames) {

  /**
    * Given a raw input data RDD, generate the partitioned unique features names grouped by hash code
    *
    * @param inputRdd
    * @return RDD[(hash key, Iterable[unique feature name])]
    */
  private def partitionedUniqueFeatures(inputRdd: RDD[GenericRecord]): RDD[(Int, Iterable[String])] = {
    // Copy it to avoid serialization of the entire class
    val fieldNamesRef = fieldNames
    val keyedFeaturesRDD = inputRdd.flatMap { r: GenericRecord =>
      // Step 1: extract feature names
      val fs = r.get(fieldNamesRef.features).asInstanceOf[JList[_]]
      val it = fs.iterator()

      val res = new Array[String](fs.size())
      var i = 0
      while (it.hasNext) {
        val record = it.next().asInstanceOf[GenericRecord]
        res(i) = Utils.getFeatureKey(record, fieldNamesRef.name, fieldNamesRef.term, GLMSuite.DELIMITER)
        i += 1
      }
      res
    }.mapPartitions{iter =>
      // Step 2. map features to (hashCode, featureName)
      val set = mutable.HashSet[String]()
      while (iter.hasNext) {
        set += iter.next()
      }
      set.toList.map(f => (f.hashCode, f)).iterator
    }

    val keyedFeaturesUnionedRDD = if (addIntercept) {
      val interceptRDD = sc.parallelize(List[(Int, String)](
          GLMSuite.INTERCEPT_NAME_TERM.hashCode() -> GLMSuite.INTERCEPT_NAME_TERM))
      keyedFeaturesRDD.union(interceptRDD)
    } else {
      keyedFeaturesRDD
    }

    // Step 3. distinct and group by hashcode
    // (note: integer's hashcode is still itself, this trick saves shuffle data size)
    keyedFeaturesUnionedRDD.distinct().groupByKey(new HashPartitioner(partitionNum))
  }

  private def buildIndexMap(featuresRdd: RDD[(Int, Iterable[String])]): Unit = {
    // Copy variable to avoid serializing the job class
    val outputPathCopy = outputPath

    val projectRdd = featuresRdd.mapPartitionsWithIndex{ case (idx, iter) =>
      // Note: PalDB writer within the same JVM might stomp on each other and generate corrupted data, it's safer to
      // lock the write. This will only block writing operations within the same JVM
      PalDBIndexMapBuilder.WRITER_LOCK.synchronized {
        val mapBuilder = new PalDBIndexMapBuilder().init(outputPathCopy, idx)

        var i: Int = 0
        while (iter.hasNext) {
          val it2 = iter.next._2.iterator
          while (it2.hasNext) {
            mapBuilder.put(it2.next(), i)
            i += 1
          }
        }

        println(s"Partition [${idx}] total record number: ${i}")
        mapBuilder.close()
      }
      iter
    }

    // Trigger run
    projectRdd.count
  }

  def run(): Unit = {
    val inputRdd = AvroIOUtils.readFromAvro[GenericRecord](sc, inputPath, 10)
    val featuresRdd = partitionedUniqueFeatures(inputRdd)
    buildIndexMap(featuresRdd)
  }
}

object FeatureIndexingJob {
  def main(args: Array[String]): Unit = {
    val sc: SparkContext = SparkContextConfiguration.asYarnClient(new SparkConf(), "build-feature-index-map", true)

    if (args.length < 3 || args.length > 5) {
      throw new IllegalArgumentException("Excepting 3-5 arguments for this job: " +
          "[input_data_path] [partition_num] [output_dir] optional: [if_add_intercept] (by default is true)" +
          "optional:[data_format](default: TRAINING_EXAMPLE, options: [TRAINING_EXAMPLE|RESPONSE_PREDICTION]"
      )
    }
    val inputPath = args(0)
    val partitionNum = args(1).toInt
    val outputPath = args(2)

    val addIntercept = if (args.length > 3) args(3).toBoolean else true
    val fieldNames: FieldNames = if (args.length > 4) {
      val fieldNamesType = FieldNamesType.withName(args(4).toUpperCase())
      fieldNamesType match {
        case RESPONSE_PREDICTION => ResponsePredictionFieldNames
        case TRAINING_EXAMPLE => TrainingExampleFieldNames
        case _ => throw new IllegalArgumentException(
            s"Input training file's field name type cannot be ${fieldNamesType}")
      }
    } else {
      TrainingExampleFieldNames
    }

    new FeatureIndexingJob(sc, inputPath, partitionNum, outputPath, addIntercept, fieldNames).run()
  }
}
