package com.linkedin.photon.ml.cli.game.scoring

/**
 * @author xazhang
 */
case class ScoredItem(ids: Iterable[String], score: Double, label: Double) {
  override def toString = s"${ids.mkString("\t")}\t$score\t$label\t"
}
