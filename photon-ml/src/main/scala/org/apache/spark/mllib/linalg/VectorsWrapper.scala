package org.apache.spark.mllib.linalg
import breeze.linalg.{Vector => BV}

/**
 * This object is a wrapper to convert mllib vectors from/to breeze vectors. Due to the constraint from the mllib
 * package, the converters have very restricted access. This class bridges the gap so the converter becomes generally
 * available.
 *
 * @author dpeng
 */
object VectorsWrapper {
  def breezeToMllib(breezeVector: BV[Double]): Vector = {
    Vectors.fromBreeze(breezeVector)
  }

  def mllibToBreeze(mllibVector: Vector): BV[Double] = {
    mllibVector.toBreeze
  }
}
