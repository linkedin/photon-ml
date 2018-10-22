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
package com.linkedin.photon.ml

/**
 * Some types that make the code easier to read and more documented. This class should be visible from everywhere in
 * photon-ml.
 */
object Types {

  type SDV = org.apache.spark.ml.linalg.DenseVector
  type SSV = org.apache.spark.ml.linalg.SparseVector
  type SparkVector = org.apache.spark.ml.linalg.Vector

  // A "sample" is a training, validation or scoring vector of feature values
  type UniqueSampleId = Long

  // A CoordinateId corresponds to a fixed effect model, random effect model, or matrix factorization model
  // e.g. "per-user", "per-song"
  type CoordinateId = String

  // RE stands for "random effect"
  // A random effect model has a single random effect type
  type REType = String

  // REId is a specific, per-item, instance of a random effect model e.g. "user123", "user456" for "per-user"
  type REId = String

  // A "feature shard" is an arbitrary set of "feature bags"
  // A random effect model corresponds to a single feature shard
  type FeatureShardId = String
}
