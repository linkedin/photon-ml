package com.linkedin.photon.ml.optimization

import com.linkedin.photon.ml.data.LabeledPoint
import com.linkedin.photon.ml.function.{TwiceDiffFunction, DiffFunction}
import com.linkedin.photon.ml.optimization.OptimizerType.OptimizerType
import org.apache.spark.Logging

protected[ml] object OptimizerFactory {

  @Deprecated
  def getOptimizer[F <: DiffFunction[LabeledPoint]](optimizerType:OptimizerType)(implicit ev:Optimizable[F]) {
    ev.getOptimizer(optimizerType)
    throw new RuntimeException(s"This method is deprecated. No one should ever trigger this. See OFFREL-934 for why.")
  }

  trait Optimizable[-T <: DiffFunction[LabeledPoint]] {
    def getOptimizer(optimizerType: OptimizerType): Optimizer[LabeledPoint, T]
  }

  object Optimizable {
    implicit object OptimizableDiffFunction extends Optimizable[DiffFunction[LabeledPoint]] with Logging {
      def getOptimizer(optimizerType:OptimizerType): Optimizer[LabeledPoint, DiffFunction[LabeledPoint]] = {
        logWarning(s"You requested an optimizer of type $optimizerType; however, right now only LBFGS is supported.")
        new LBFGS[LabeledPoint]()
      }
    }

    implicit object OptimizableTwiceDiffFunction extends Optimizable[TwiceDiffFunction[LabeledPoint]] {
      def getOptimizer(
          optimizerType:OptimizerType): Optimizer[LabeledPoint, TwiceDiffFunction[LabeledPoint]] =

        optimizerType match {
          case OptimizerType.LBFGS => new LBFGS[LabeledPoint]()
          case OptimizerType.TRON => new TRON[LabeledPoint]()
          case _ => throw new IllegalArgumentException(s"Don't know how to deal with optimizer type ${optimizerType}")
      }
    }
  }
}
