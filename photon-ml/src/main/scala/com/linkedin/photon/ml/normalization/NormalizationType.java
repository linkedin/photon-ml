package com.linkedin.photon.ml.normalization;

/**
 * The enum of different normalization types used for feature normalization.
 * @author dpeng
 */
public enum NormalizationType {

  /**
   * Scale each feature to have unit variance
   */
  USE_STANDARD_DEVIATION,

  /**
   * Scale each feature to lie in the range [-1, 1]
   */
  USE_MAX_MAGNITUDE,

  /**
   * Skip scaling
   */
  NO_SCALING
}
