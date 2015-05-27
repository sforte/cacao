package utils

import breeze.linalg.DenseVector
import models.{Model,RealFunction, Loss, Regularizer}
import org.apache.spark.rdd.RDD
import utils.OptUtils._
import vectors.LabelledPoint

trait ConvergenceChecker[-L<:Loss[_,_]] extends Serializable {

  def hasConverged (
    model: Model[L],
    data: RDD[Array[LabelledPoint]],
    alphaInit: RDD[DenseVector[Double]],
    v: DenseVector[Double],
    round: Int
  ): Boolean
}

class DualityGapConvergenceChecker[-L<:Loss[RealFunction,RealFunction]]
  (epsilon: Double = 10E-8, maxRounds: Int = 1000, verbose: Boolean = true)
    extends ConvergenceChecker[L] {

  def hasConverged (
    model: Model[L],
    data: RDD[Array[LabelledPoint]],
    alpha: RDD[DenseVector[Double]],
    v: DenseVector[Double],
    round: Int
  ) = {

    val primalObj = computePrimalObjective(data, model, model.regularizer.dualGradient(v))
    val dualObj = computeDualObjective(data, model, v, alpha)
    val gap = primalObj - dualObj

    if (verbose) {
      println(
        s"Iteration: $round\t",
        s"Objective Value: $primalObj\t",
        s"Dual Objective Value: $dualObj\t",
        s"Duality Gap: $gap\t"
      )
    }

    round > maxRounds || gap < epsilon
  }
}
