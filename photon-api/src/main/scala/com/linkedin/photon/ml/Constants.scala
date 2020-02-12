package com.linkedin.photon.ml

import org.joda.time.DateTimeZone

import com.linkedin.photon.ml.util.Utils

/**
 * Some commonly used String constants.
 */
object Constants {

  /**
   * Delimiter used to concatenate feature name and term into feature key.
   *
   * WARNING: This is not visible in println!
   */
  val DELIMITER = "\u0001"

  /**
   * Wildcard character used for specifying the feature constraints. Only the term is allowed to be a wildcard normally
   * unless one wants to apply bounds to all features in which case both name and term can be specified as wildcards.
   * Currently, we do not support wildcards in name alone.
   */
  val WILDCARD = "*"

  val INTERCEPT_NAME = "(INTERCEPT)"
  val INTERCEPT_TERM = ""
  val INTERCEPT_KEY = Utils.getFeatureKey(INTERCEPT_NAME, INTERCEPT_TERM)

  /**
   * Default time zone for relative date calculations
   */
  val DEFAULT_TIME_ZONE = DateTimeZone.UTC

  val UNIQUE_SAMPLE_ID = "uniqueId"
}