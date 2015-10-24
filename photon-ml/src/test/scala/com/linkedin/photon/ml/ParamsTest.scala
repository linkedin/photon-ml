package com.linkedin.photon.ml

import com.linkedin.photon.ml.io.FieldNamesType
import FieldNamesType.FieldNamesType
import com.linkedin.photon.ml.optimization.{OptimizerType, RegularizationType}
import OptimizerType.OptimizerType
import RegularizationType.RegularizationType
import com.linkedin.photon.ml.optimization.RegularizationType
import com.linkedin.photon.ml.io.FieldNamesType
import com.linkedin.photon.ml.normalization.NormalizationType
import com.linkedin.photon.ml.optimization.RegularizationType

import org.testng.Assert._
import org.testng.annotations.Test

/**
 * Merely tests the apply, unapply method of Params case class
 * Note: this test is merely for covering something almost for sure,
 *     since case class is automatically interpreted by Scala compilers.
 *
 * @author yizhou
 * @author dpeng
 */
class ParamsTest {

  @Test
  def testDefaultApplyAndUnapply(): Unit = {
    val params = Params()
    assertNull(params.trainDir)
    assertEquals(params.validateDirOpt, None)
    assertNull(params.outputDir)
    assertNull(params.taskType)
    assertEquals(params.maxNumIter, 80)
    assertEquals(params.regularizationWeights, List(0.1, 1, 10, 100))
    assertEquals(params.tolerance, 1e-6)
    assertEquals(params.optimizerType, OptimizerType.LBFGS)
    assertEquals(params.regularizationType, RegularizationType.L2)
    assertTrue(params.addIntercept)
    assertTrue(params.enableOptimizationStateTracker)
    assertFalse(params.validatePerIteration)
    assertEquals(params.minNumPartitions, 1)
    assertTrue(params.kryo)
    assertEquals(params.fieldsNameType, FieldNamesType.RESPONSE_PREDICTION)
    assertEquals(params.summarizationOutputDirOpt, None)
    assertEquals(params.normalizationType, NormalizationType.NONE)
    assertEquals(params.jobName, s"Photon-ML-Training")
    assertEquals(params.constraintString, None)

    val Params(_,
        validateDirOpt: Option[String],
        _,
        _,
        maxNumIter: Int,
        regularizationWeights: List[Double],
        tolerance: Double,
        optimizerType: OptimizerType,
        regularizationType: RegularizationType,
        alphaOption: Option[Double],
        addIntercept: Boolean,
        enableOptimizationStateTracker: Boolean,
        validatePerIteration: Boolean,
        minNumPartitions: Int,
        kryo: Boolean,
        fieldsNameType: FieldNamesType,
        summarizationOutputDirOpt: Option[String],
        normalizationType: NormalizationType,
        jobName: String,
        constraintString: Option[String]) = params

    assertEquals(validateDirOpt, None)
    assertEquals(maxNumIter, 80)
    assertEquals(regularizationWeights, List(0.1, 1, 10, 100))
    assertEquals(tolerance, 1e-6)
    assertEquals(optimizerType, OptimizerType.LBFGS)
    assertEquals(regularizationType, RegularizationType.L2)
    assertTrue(addIntercept)
    assertTrue(enableOptimizationStateTracker)
    assertFalse(validatePerIteration)
    assertEquals(minNumPartitions, 1)
    assertTrue(kryo)
    assertEquals(fieldsNameType, FieldNamesType.RESPONSE_PREDICTION)
    assertEquals(summarizationOutputDirOpt, None)
    assertEquals(normalizationType, NormalizationType.NONE)
    assertEquals(jobName, s"Photon-ML-Training")
    assertEquals(constraintString, None)
  }

  @Test
  def testEquals(): Unit = {
    val regWeights1 = List(0.1, 0.2, 1.0)
    val regWeights2 = List(0.1, 0.2, 1.0)
    assertEquals(Params(regularizationWeights = regWeights1), Params(regularizationWeights = regWeights2))
    assertNotEquals(Params(regularizationWeights = regWeights1), Params(trainDir = "foo/bar/tar",
        regularizationWeights = regWeights2))
    assertEquals(Params(trainDir = "foo/bar/tar", regularizationWeights = regWeights1), Params(trainDir = "foo/bar/tar",
        regularizationWeights = regWeights2))

    assertEquals(Params(regularizationWeights = regWeights1).hashCode(),
        Params(regularizationWeights = regWeights2).hashCode())
    assertNotEquals(Params(regularizationWeights = regWeights1).hashCode(), Params(trainDir = "foo/bar/tar",
        regularizationWeights = regWeights2).hashCode())
    assertEquals(Params(trainDir = "foo/bar/tar", regularizationWeights = regWeights1).hashCode(),
        Params(trainDir = "foo/bar/tar", regularizationWeights = regWeights2).hashCode())
  }
}
