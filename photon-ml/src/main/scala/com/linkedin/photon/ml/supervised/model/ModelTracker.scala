package com.linkedin.photon.ml.supervised.model

import com.linkedin.photon.ml.optimization.OptimizationStatesTracker
import scala.language.existentials

// existentials is imported to suppress the warning message:
// photon_trunk/photon-ml/src/main/scala/com/linkedin/photon/ml/supervised/model/ModelTracker.scala:11: inferred
//   existential type Option[(com.linkedin.photon.ml.optimization.OptimizationStatesTracker, Array[_$1])] forSome
//   { type _$1 <: com.linkedin.photon.ml.supervised.model.GeneralizedLinearModel }, which cannot be expressed by
//   wildcards,  should be enabled
// by making the implicit value scala.language.existentials visible.
// This can be achieved by adding the import clause 'import scala.language.existentials'
// or by setting the compiler option -language:existentials.
// See the Scala docs for value scala.language.existentials for a discussion
// why the feature should be explicitly enabled.

/**
 * A model tracker to include optimization state and per iteration models.
 * @author dpeng
 */
case class ModelTracker(
  optimizationStateTrackerString: String,
  models: Array[_ <: GeneralizedLinearModel])
