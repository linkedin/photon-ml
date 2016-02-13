package com.linkedin.photon.ml.test

import breeze.linalg.Vector
import breeze.linalg.DenseVector

import scala.collection.mutable.ListBuffer
import scala.util.Random

/**
 * A collection of handy utils useful in tests
 *
 * @author yizhou
 */
object CommonTestUtils {

  /**
   * Append prefix to a CMD line option name, forming an argument string
   *
   * @param optionName the option name
   * @return the argument string
   */
  def fromOptionNameToArg(optionName: String): String = "--" + optionName

  /**
   * This a utility for comparing two constraint maps in unit tests. Returns true if both the options are None or are
   * non-empty and the contained maps have the same set of (key, value) pairs
   *
   * @param m1 option containing the first map
   * @param m2 option containing the second map
   * @return true iff both options are none or contain maps that have the exact same set of (key, value) tuples
   */
  def compareConstraintMaps(m1: Option[Map[Int, (Double, Double)]], m2: Option[Map[Int, (Double, Double)]]): Boolean = {
    (m1, m2) match {
      case (Some(x), Some(y)) =>
        if (x.size != y.size) {
          false
        } else {
          x.foreach{
            case (id, bounds) =>
              y.get(id) match {
                case Some(w: (Double, Double)) =>
                  if (w != bounds) {
                    false
                  }
                case None => false
              }
          }
          true
        }
      case (None, None) => true
      case (None, _) => false
      case (_, None) => false
    }
  }

  /**
   * Generates given number of valid and invalid dense feature vectors of given dimension
   *
   * @param numValidVectors number of valid vectors to generate
   * @param numInvalidVectors number of invalid vectors to generate
   * @param numDimensions the dimension of the generated feature vectors
   * @return A sequence of generated dense feature vectors
   */
  def generateDenseFeatureVectors(numValidVectors: Int, numInvalidVectors: Int,
                             numDimensions: Int): Seq[Vector[Double]] = {
    val r = new Random(System.currentTimeMillis())
    val result = new ListBuffer[Vector[Double]]

    for (i <- 0 until numValidVectors) {
      val v = new Array[Double](numDimensions)
      for (j <- 0 until numDimensions) {
        v(j) = r.nextDouble()
      }
      result +=  new DenseVector[Double](v)
    }

    for (i <- 0 until numInvalidVectors) {
      val v = new Array[Double](numDimensions)
      /* In the worst case, if all our coin tosses below fail, we might end up with a valid vector.
         Hence setting the first element invalid to guarantee an invalid vector */
      v(0) = Double.NaN
      for (j <- 1 until numDimensions) {
        v(j) = r.nextBoolean() match {
          case true =>
            r.nextInt(3) match {
              case 0 => Double.NaN
              case 1 => Double.PositiveInfinity
              case 2 => Double.NegativeInfinity
            }
          case false => r.nextDouble()
        }
      }
      result += new DenseVector[Double](v)
    }

    result
  }
}