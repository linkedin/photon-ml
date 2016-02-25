package com.linkedin.photon.ml.function

import java.util.Random

import breeze.linalg.{DenseVector, SparseVector, Vector}
import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.optimization._
import com.linkedin.photon.ml.test.SparkTestUtils
import org.apache.log4j.{LogManager, Logger}
import org.apache.spark.broadcast.Broadcast
import org.testng.Assert.{assertEquals, assertTrue}
import org.testng.annotations.{DataProvider, Test}

import scala.collection.mutable.ListBuffer

/**
 * Unit tests to verify functions have gradients / Hessians to be consistent with each other.
 * @author bdrew
 * @author yali
 */
class ObjectiveFunctionIntegTest extends SparkTestUtils {

  /**
   * List everything that conforms to DiffFunction here
   */
  @DataProvider(parallel = true)
  def getDifferentiableFunctions: Array[Array[Any]] = {
    // List of functions that return a tuple containing an undecorated loss function and its corresponding local data
    // set
    val baseLossFunctions = Array(
      () => ("Differentiable dummy objective to test optimizer, benign data", new TestObjective(), generateBenignLocalDataSetBinaryClassification()),
      () => ("Differentiable logistic loss, benign data", new LogisticLossFunction(), generateBenignLocalDataSetBinaryClassification()),
      () => ("Differentiable smoothed hinge loss, benign data", new SmoothedHingeLossFunction(), generateBenignLocalDataSetBinaryClassification()),
      () => ("Differentiable squared loss, benign data", new SquaredLossFunction(), generateBenignLocalDataSetLinearRegression()),
      () => ("Differentiable poisson loss, benign data", new PoissonLossFunction(), generateBenignLocalDataSetPoissonRegression()),
      () => ("Differentiable dummy objective to test optimizer, outlier data", new TestObjective(), generateBenignLocalDataSetBinaryClassification()),
      () => ("Differentiable logistic loss, outlier data", new LogisticLossFunction(), generateOutlierLocalDataSetBinaryClassification()),
      () => ("Differentiable smoothed hinge loss, outlier data", new SmoothedHingeLossFunction(), generateBenignLocalDataSetBinaryClassification()),
      () => ("Differentiable squared loss, outlier data", new SquaredLossFunction(), generateOutlierLocalDataSetLinearRegression()),
      () => ("Differentiable poisson loss, outlier data", new PoissonLossFunction(), generateOutlierLocalDataSetPoissonRegression()))

    (for {base <- baseLossFunctions} yield {
      val (desc, undecorated, data) = base()
      val diffAdapted = undecorated match {
        case df: DiffFunction[LabeledPoint] =>
          Seq(
            (desc, df, data),
            (s"$desc with diff function L2 regularization", DiffFunction.withRegularization(df, L2RegularizationContext, ObjectiveFunctionTest.REGULARIZATION_STRENGTH), data),
            (s"$desc with diff function L1 regularization", DiffFunction.withRegularization(df, L1RegularizationContext, ObjectiveFunctionTest.REGULARIZATION_STRENGTH), data))
        case _ => Seq()
      }

      val twiceDiffAdapted = undecorated match {
        case twiceDF: TwiceDiffFunction[LabeledPoint] =>
          Seq(
            (s"$desc with twice diff function L2 regularization", TwiceDiffFunction.withRegularization(twiceDF, L2RegularizationContext, ObjectiveFunctionTest.REGULARIZATION_STRENGTH), data),
            (s"$desc with twice diff function L1 regularization", TwiceDiffFunction.withRegularization(twiceDF, L1RegularizationContext, ObjectiveFunctionTest.REGULARIZATION_STRENGTH), data)
          )

        case _ => Seq()
      }

      diffAdapted ++ twiceDiffAdapted
    }).flatMap(_.iterator).flatMap(x => Array(Array(x._1, x._2, x._3, 1), Array(x._1, x._2, x._3, 2))).toArray
  }


  /**
   * List everything that conforms to TwiceDiffFunction here
   */
  @DataProvider(parallel = true)
  def getTwiceDifferentiableFunctions: Array[Array[Any]] = {
    val baseLossFunctions = Array(
      () => ("Differentiable logistic loss, benign data", new LogisticLossFunction(), generateBenignLocalDataSetBinaryClassification()),
      () => ("Differentiable squared loss, benign data", new SquaredLossFunction(), generateBenignLocalDataSetLinearRegression()),
      () => ("Differentiable poisson loss, benign data", new PoissonLossFunction(), generateBenignLocalDataSetPoissonRegression()),
      () => ("Differentiable logistic loss, outlier data", new LogisticLossFunction(), generateOutlierLocalDataSetBinaryClassification()),
      () => ("Differentiable squared loss, outlier data", new SquaredLossFunction(), generateOutlierLocalDataSetLinearRegression()),
      () => ("Differentiable poisson loss, outlier data", new PoissonLossFunction(), generateOutlierLocalDataSetPoissonRegression()))

    // List of regularization decorators. For each item in the base loss function list, we apply each decorator
    val regularizationDecorators = Array(
      (x: TwiceDiffFunction[LabeledPoint], baseDesc: String) => (s"$baseDesc with TwiceDiffFunction L2 regularization", TwiceDiffFunction.withRegularization(x, L2RegularizationContext, ObjectiveFunctionIntegTest.REGULARIZATION_STRENGTH))
    )

    val tmp: scala.collection.mutable.ListBuffer[(String, DiffFunction[LabeledPoint], Seq[LabeledPoint])] = scala.collection.mutable.ListBuffer()

    // Generate cartesian product of all regularization types by all base loss functions
    baseLossFunctions.map({ f =>
      val undecorated = f()
      tmp.append(undecorated)

      regularizationDecorators.map({ regularize =>
        val decorated = regularize(undecorated._2, undecorated._1)
        tmp.append((decorated._1, decorated._2, undecorated._3))
      })
    })

    tmp.flatMap(x => Array(Array(x._1, x._2, x._3, 1), Array(x._1, x._2, x._3, 2))).toArray
  }

  /**
   * Generate benign datasets with weights. The output contains 2 * TRAINING_SAMPLES
   * examples in total. Have of them have weight = 1, and have of them have random weights.
   * @return a list of [[LabeledPoint]]
   */
  def generateBenignLocalDataSetBinaryClassification(): List[LabeledPoint] = {
    val tmp1: List[LabeledPoint] = drawBalancedSampleFromNumericallyBenignDenseFeaturesForBinaryClassifierLocal(
      ObjectiveFunctionIntegTest.DATA_RANDOM_SEED,
      ObjectiveFunctionIntegTest.TRAINING_SAMPLES,
      ObjectiveFunctionIntegTest.PROBLEM_DIMENSION).map({ obj =>
      assertEquals(obj._2.length, ObjectiveFunctionTest.PROBLEM_DIMENSION, "Samples should have expected lengths")
      new LabeledPoint(label = obj._1, obj._2, offset = 0, weight = 1)
    }).toList
    //generate random points with random weights
    val r: Random = new Random(ObjectiveFunctionTest.WEIGHT_RANDOM_SEED)
    val tmp2: List[LabeledPoint] = drawBalancedSampleFromNumericallyBenignDenseFeaturesForBinaryClassifierLocal(
      ObjectiveFunctionIntegTest.DATA_RANDOM_SEED,
      ObjectiveFunctionIntegTest.TRAINING_SAMPLES,
      ObjectiveFunctionIntegTest.PROBLEM_DIMENSION).map({ obj =>
      assertEquals(obj._2.length, ObjectiveFunctionTest.PROBLEM_DIMENSION, "Samples should have expected lengths")
      val weight: Double = r.nextDouble() * ObjectiveFunctionTest.WEIGHT_RANDOM_MAX
      new LabeledPoint(label = obj._1, obj._2, offset = 0, weight)
    }).toList
    tmp1 ::: tmp2
  }

  /**
   * Generate outliers with weights. The output contains 2 * TRAINING_SAMPLES
   * examples in total. Have of them have weight = 1, and have of them have random weights.
   * @return a list of [[LabeledPoint]]
   */
  def generateOutlierLocalDataSetBinaryClassification(): List[LabeledPoint] = {
    val tmp1: List[LabeledPoint] = drawBalancedSampleFromOutlierDenseFeaturesForBinaryClassifierLocal(
      ObjectiveFunctionIntegTest.DATA_RANDOM_SEED,
      ObjectiveFunctionIntegTest.TRAINING_SAMPLES,
      ObjectiveFunctionIntegTest.PROBLEM_DIMENSION).map({ obj =>
      assertEquals(obj._2.length, ObjectiveFunctionIntegTest.PROBLEM_DIMENSION, "Samples should have expected lengths")
      new LabeledPoint(label = obj._1, obj._2, offset = 0, weight = 1)
    }).toList

    //generate random points with random weights
    val r: Random = new Random(ObjectiveFunctionTest.WEIGHT_RANDOM_SEED)
    val tmp2: List[LabeledPoint] = drawBalancedSampleFromOutlierDenseFeaturesForBinaryClassifierLocal(
      ObjectiveFunctionIntegTest.DATA_RANDOM_SEED,
      ObjectiveFunctionIntegTest.TRAINING_SAMPLES,
      ObjectiveFunctionIntegTest.PROBLEM_DIMENSION).map({ obj =>
      assertEquals(obj._2.length, ObjectiveFunctionIntegTest.PROBLEM_DIMENSION, "Samples should have expected lengths")
      val weight: Double = r.nextDouble() * ObjectiveFunctionTest.WEIGHT_RANDOM_MAX
      new LabeledPoint(label = obj._1, obj._2, offset = 0, weight)
    }).toList
    tmp1 ::: tmp2
  }

  def generateBenignLocalDataSetPoissonRegression(): List[LabeledPoint] = {
    val tmp1: List[LabeledPoint] = drawSampleFromNumericallyBenignDenseFeaturesForPoissonRegressionLocal(
      ObjectiveFunctionIntegTest.DATA_RANDOM_SEED,
      ObjectiveFunctionIntegTest.TRAINING_SAMPLES,
      ObjectiveFunctionIntegTest.PROBLEM_DIMENSION).map({ obj =>
      assertEquals(obj._2.length, ObjectiveFunctionIntegTest.PROBLEM_DIMENSION, "Samples should have expected lengths")
      new LabeledPoint(label = obj._1, obj._2, offset = 0, weight = 1)
    }).toList

    //generate random points with random weights
    val r: Random = new Random(ObjectiveFunctionTest.WEIGHT_RANDOM_SEED)
    val tmp2: List[LabeledPoint] = drawSampleFromNumericallyBenignDenseFeaturesForPoissonRegressionLocal(
      ObjectiveFunctionIntegTest.DATA_RANDOM_SEED,
      ObjectiveFunctionIntegTest.TRAINING_SAMPLES,
      ObjectiveFunctionIntegTest.PROBLEM_DIMENSION).map({ obj =>
      assertEquals(obj._2.length, ObjectiveFunctionIntegTest.PROBLEM_DIMENSION, "Samples should have expected lengths")
      val weight: Double = r.nextDouble() * ObjectiveFunctionTest.WEIGHT_RANDOM_MAX
      new LabeledPoint(label = obj._1, obj._2, offset = 0, weight)
    }).toList
    tmp1 ::: tmp2
  }

  def generateOutlierLocalDataSetPoissonRegression(): List[LabeledPoint] = {
    val tmp1: List[LabeledPoint] = drawSampleFromOutlierDenseFeaturesForPoissonRegressionLocal(
      ObjectiveFunctionIntegTest.DATA_RANDOM_SEED,
      ObjectiveFunctionIntegTest.TRAINING_SAMPLES,
      ObjectiveFunctionIntegTest.PROBLEM_DIMENSION).map({ obj =>
      assertEquals(obj._2.length, ObjectiveFunctionIntegTest.PROBLEM_DIMENSION, "Samples should have expected lengths")
      new LabeledPoint(label = obj._1, obj._2, offset = 0, weight = 1)
    }).toList

    //generate random points with random weights
    val r: Random = new Random(ObjectiveFunctionTest.WEIGHT_RANDOM_SEED)
    val tmp2: List[LabeledPoint] = drawSampleFromOutlierDenseFeaturesForPoissonRegressionLocal(
      ObjectiveFunctionIntegTest.DATA_RANDOM_SEED,
      ObjectiveFunctionIntegTest.TRAINING_SAMPLES,
      ObjectiveFunctionIntegTest.PROBLEM_DIMENSION).map({ obj =>
      assertEquals(obj._2.length, ObjectiveFunctionIntegTest.PROBLEM_DIMENSION, "Samples should have expected lengths")
      val weight: Double = r.nextDouble() * ObjectiveFunctionTest.WEIGHT_RANDOM_MAX
      new LabeledPoint(label = obj._1, obj._2, offset = 0, weight)
    }).toList
    tmp1 ::: tmp2
  }

  def generateBenignLocalDataSetLinearRegression(): List[LabeledPoint] = {
    val tmp1: List[LabeledPoint] = drawSampleFromNumericallyBenignDenseFeaturesForLinearRegressionLocal(
      ObjectiveFunctionIntegTest.DATA_RANDOM_SEED,
      ObjectiveFunctionIntegTest.TRAINING_SAMPLES,
      ObjectiveFunctionIntegTest.PROBLEM_DIMENSION).map({ obj =>
      assertEquals(obj._2.length, ObjectiveFunctionIntegTest.PROBLEM_DIMENSION, "Samples should have expected lengths")
      new LabeledPoint(label = obj._1, obj._2, offset = 0, weight = 1)
    }).toList

    //generate random points with random weights
    val r: Random = new Random(ObjectiveFunctionTest.WEIGHT_RANDOM_SEED)
    val tmp2: List[LabeledPoint] = drawSampleFromNumericallyBenignDenseFeaturesForLinearRegressionLocal(
      ObjectiveFunctionIntegTest.DATA_RANDOM_SEED,
      ObjectiveFunctionIntegTest.TRAINING_SAMPLES,
      ObjectiveFunctionIntegTest.PROBLEM_DIMENSION).map({ obj =>
      assertEquals(obj._2.length, ObjectiveFunctionIntegTest.PROBLEM_DIMENSION, "Samples should have expected lengths")
      val weight: Double = r.nextDouble() * ObjectiveFunctionTest.WEIGHT_RANDOM_MAX
      new LabeledPoint(label = obj._1, obj._2, offset = 0, weight)
    }).toList
    tmp1 ::: tmp2
  }

  def generateOutlierLocalDataSetLinearRegression(): List[LabeledPoint] = {
    val tmp1: List[LabeledPoint] = drawSampleFromOutlierDenseFeaturesForLinearRegressionLocal(
      ObjectiveFunctionIntegTest.DATA_RANDOM_SEED,
      ObjectiveFunctionIntegTest.TRAINING_SAMPLES,
      ObjectiveFunctionIntegTest.PROBLEM_DIMENSION).map({ obj =>
      assertEquals(obj._2.length, ObjectiveFunctionIntegTest.PROBLEM_DIMENSION, "Samples should have expected lengths")
      new LabeledPoint(label = obj._1, obj._2, offset = 0, weight = 1)
    }).toList

    //generate random points with random weights
    val r: Random = new Random(ObjectiveFunctionTest.WEIGHT_RANDOM_SEED)
    val tmp2: List[LabeledPoint] = drawSampleFromOutlierDenseFeaturesForLinearRegressionLocal(
      ObjectiveFunctionIntegTest.DATA_RANDOM_SEED,
      ObjectiveFunctionIntegTest.TRAINING_SAMPLES,
      ObjectiveFunctionIntegTest.PROBLEM_DIMENSION).map({ obj =>
      assertEquals(obj._2.length, ObjectiveFunctionIntegTest.PROBLEM_DIMENSION, "Samples should have expected lengths")
      val weight: Double = r.nextDouble() * ObjectiveFunctionTest.WEIGHT_RANDOM_MAX
      new LabeledPoint(label = obj._1, obj._2, offset = 0, weight)
    }).toList
    tmp1 ::: tmp2
  }

  def checkGradient(prefix: String, objectiveBefore: Double, objectiveAfter: Double, expected: Double) = {
    val numericDeriv = (objectiveAfter - objectiveBefore) / (2.0 * ObjectiveFunctionIntegTest.DERIVATIVE_DELTA)
    val relativeErrorNum = Math.abs(numericDeriv - expected)
    val relativeErrorFactor = Math.min(Math.abs(numericDeriv), Math.abs(expected))
    val relativeErrorDen = if (relativeErrorFactor > 0) {
      relativeErrorFactor
    } else {
      1
    }
    val relativeError = relativeErrorNum / relativeErrorDen

    assertTrue(java.lang.Double.isFinite(objectiveBefore), s"Objective before step [$objectiveBefore] should be finite")
    assertTrue(java.lang.Double.isFinite(objectiveAfter), s"Objective after step [$objectiveAfter] should be finite")

    if (!(numericDeriv.isInfinite || numericDeriv.isNaN)) {
      assertTrue(
        relativeError < ObjectiveFunctionIntegTest.GRADIENT_TOLERANCE ||
          relativeErrorNum < ObjectiveFunctionIntegTest.GRADIENT_TOLERANCE,
        "Computed gradient and numerical differentiation estimate should be close." +
          s"$prefix estimated [$numericDeriv] " +
          s"v. computed [$expected] with absolute error [$relativeErrorNum] and " +
          s"relative error [$relativeError]")
    }
  }

  def checkHessian(prefix: String, gradBefore: Double, gradAfter: Double, expected: Double) = {
    val numericDeriv = (gradAfter - gradBefore) / (2.0 * ObjectiveFunctionIntegTest.DERIVATIVE_DELTA)
    val relativeErrorNum = Math.abs(numericDeriv - expected)
    val relativeErrorFactor = Math.min(Math.abs(numericDeriv), Math.abs(expected))
    val relativeErrorDen = if (relativeErrorFactor > 0) {
      relativeErrorFactor
    } else {
      1
    }
    val relativeError = relativeErrorNum / relativeErrorDen

    assertTrue(
      relativeError < ObjectiveFunctionIntegTest.HESSIAN_TOLERANCE ||
        relativeErrorNum < ObjectiveFunctionIntegTest.HESSIAN_TOLERANCE,
      "Computed Hessian and numerical differentiation estimate should be close." +
        s"$prefix estimated [$numericDeriv] " +
        s"v. computed [$expected] with absolute error [$relativeErrorNum] and " +
        s"relative error [$relativeError]")
  }

  /**
   * Verify that gradient is consistent with the objective when computed via Spark
   *
   *
   * Note: here, rather than calling computeAt(...) to get both the objective and gradient
   * in one shot, we use DiffFunction#value and DiffFunction#gradient instead. This is to
   * ensure that we get some coverage of these functions which aren't used anywhere else.
   * In the near term, we should decide if we want to keep those methods as part of the
   * design or remove them, as they aren't used by any of the solvers.
   */
  @Test(dataProvider = "getDifferentiableFunctions",
    groups = Array[String]("ObjectiveFunctionTests", "testCore"))
  def checkGradientConsistentWithObjectiveSpark(description: String, function: DiffFunction[LabeledPoint], localData: Seq[LabeledPoint], treeAggregateDepth: Int): Unit = sparkTest("checkGradientConsistentWithObjectiveSpark") {
    function.treeAggregateDepth = treeAggregateDepth
    val data = sc.parallelize(localData).repartition(Runtime.getRuntime.availableProcessors)
    val r: Random = new Random(ObjectiveFunctionIntegTest.PARAMETER_RANDOM_SEED)

    for (iter <- 0 until ObjectiveFunctionIntegTest.SPARK_CONSISTENCY_CHECK_SAMPLES) {
      val initParam: Vector[Double] = DenseVector.fill[Double](ObjectiveFunctionIntegTest.PROBLEM_DIMENSION) { if (iter > 0) { r.nextDouble() } else { 0 } }
      val bcastParams: Broadcast[Vector[Double]] = sc.broadcast(initParam)
      val computed = function.gradient(data, bcastParams)

      // Element-wise numerical differentiation to get the gradient
      for (idx <- 0 until ObjectiveFunctionIntegTest.PROBLEM_DIMENSION) {
        val before = initParam.copy
        before(idx) -= ObjectiveFunctionIntegTest.DERIVATIVE_DELTA
        val after = initParam.copy
        after(idx) += ObjectiveFunctionIntegTest.DERIVATIVE_DELTA
        val objBefore = function.value(data, sc.broadcast(before))
        val objAfter = function.value(data, sc.broadcast(after))

        checkGradient(
          s" f=[$function / ${function.getClass.getName}], iter=[$iter], idx=[$idx] ",
          objBefore,
          objAfter,
          computed(idx))
      }
    }
  }

  /**
   * Verify that the Hessian is consistent with the gradient when computed via Spark
   */
  @Test(dataProvider = "getTwiceDifferentiableFunctions",
    groups = Array[String]("ObjectiveFunctionTests", "testCore"))
  def checkHessianConsistentWithObjectiveSpark(description: String, function: TwiceDiffFunction[LabeledPoint], localData: Seq[LabeledPoint], treeAggregateDepth: Int): Unit = sparkTest("checkHessianConsistentWithObjectiveSpark") {
    function.treeAggregateDepth = treeAggregateDepth
    val data = sc.parallelize(localData).repartition(Runtime.getRuntime.availableProcessors)
    val r: Random = new Random(ObjectiveFunctionIntegTest.PARAMETER_RANDOM_SEED)

    for (iter <- 0 until ObjectiveFunctionIntegTest.SPARK_CONSISTENCY_CHECK_SAMPLES) {
      val initParam: Vector[Double] = DenseVector.fill[Double](ObjectiveFunctionIntegTest.PROBLEM_DIMENSION) { if (iter > 0) { r.nextDouble() } else { 0 } }

      // Loop over basis vectors. This will give us H*e_i = H(:,i) (so one column of H at a time)
      for (basis <- 0 until ObjectiveFunctionIntegTest.PROBLEM_DIMENSION) {
        val basisVector: Vector[Double] = new SparseVector[Double](Array[Int](basis), Array[Double](1.0), 1, ObjectiveFunctionIntegTest.PROBLEM_DIMENSION)
        val hessianVector = function.hessianVector(data, sc.broadcast(initParam), sc.broadcast(basisVector))

        // Element-wise numerical differentiation to get the Hessian
        for (idx <- 0 until ObjectiveFunctionIntegTest.PROBLEM_DIMENSION) {
          val before = initParam.copy
          before(idx) -= ObjectiveFunctionIntegTest.DERIVATIVE_DELTA
          val after = initParam.copy
          after(idx) += ObjectiveFunctionIntegTest.DERIVATIVE_DELTA
          val gradBefore = function.gradient(data, sc.broadcast(before))
          val gradAfter = function.gradient(data, sc.broadcast(after))
          checkHessian(
            s"Iteration [$iter], basis=[$basis], idx=[$idx], Hessian=[$hessianVector]",
            gradBefore(basis),
            gradAfter(basis),
            hessianVector(idx))
        }
      }
    }
  }
}

object ObjectiveFunctionIntegTest {
  val LOCAL_CONSISTENCY_CHECK_SAMPLES = 1000
  val SPARK_CONSISTENCY_CHECK_SAMPLES = 10
  val PROBLEM_DIMENSION: Int = 10
  val REGULARIZATION_STRENGTH: Double = 100
  val DERIVATIVE_DELTA: Double = 1e-6
  val GRADIENT_TOLERANCE: Double = 1e-3
  val HESSIAN_TOLERANCE: Double = 1e-3
  val DATA_RANDOM_SEED: Int = 0
  val PARAMETER_RANDOM_SEED: Int = 500
  val WEIGHT_RANDOM_SEED = 100
  val WEIGHT_RANDOM_MAX = 10
  val TRAINING_SAMPLES = PROBLEM_DIMENSION * PROBLEM_DIMENSION
  val LOGGER: Logger = LogManager.getLogger(classOf[ObjectiveFunctionIntegTest])
}
