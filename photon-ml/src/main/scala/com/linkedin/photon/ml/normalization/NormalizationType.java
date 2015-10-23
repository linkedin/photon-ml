package com.linkedin.photon.ml.normalization;

/**
 * The enum of different normalization types used for feature normalization.
 * @author dpeng
 */
public enum NormalizationType {

  /**
   * Scale each feature to have unit variance
   */
  SCALE_WITH_STANDARD_DEVIATION,

  /**
   * Scale each feature to lie in the range [-1, 1]
   */
  SCALE_WITH_MAX_MAGNITUDE,

  /**
   * Zero-mean unit variance distributions x -> (x - \mu)/\sigma. Intercept must be included to enable this feature.
   */
  STANDARDIZATION,

  /**
   * Skip normalization
   */
  NONE
}
