package com.linkedin.photon.ml.avro.scoring

/**
 * @author xazhang
 */
case class ScoredItem(ids: Iterable[String], score: Double, label: Double) {
  override def toString = s"${ids.mkString("\t")}\t$score\t$label\t"
}
