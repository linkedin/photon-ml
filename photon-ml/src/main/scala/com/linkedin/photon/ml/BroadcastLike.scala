package com.linkedin.photon.ml

/**
 * A trait to hold some simple operations on the Broadcasted variables
 * @author xazhang
 */
trait BroadcastLike {
  /**
   * Asynchronously delete cached copies of this broadcast on the executors
   * @return This object with all its broadcasted variables unpersisted
   */
  def unpersistBroadcast(): this.type
}
