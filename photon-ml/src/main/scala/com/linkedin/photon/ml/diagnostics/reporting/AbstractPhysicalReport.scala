package com.linkedin.photon.ml.diagnostics.reporting

import java.util.concurrent.atomic.AtomicLong

/**
 * Created by bdrew on 10/9/15.
 */
class AbstractPhysicalReport extends PhysicalReport {
  private val id:Long = AbstractPhysicalReport.GLOBAL_ID_COUNTER.getAndIncrement()
  def getId():Long = id
}

object AbstractPhysicalReport {
  val GLOBAL_ID_COUNTER:AtomicLong = new AtomicLong(0L)
}
