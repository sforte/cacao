package utils

import breeze.linalg.DenseVector
import models.{RealFunction, Loss, Regularizer}
import org.apache.spark.rdd.RDD
import utils.OptUtils._
import vectors.LabelledPoint

trait ConvergenceChecker extends Serializable {

  def hasConverged(
    loss: Loss[_<:RealFunction, _<:RealFunction],
    regularizer: Regularizer,
    n: Long,
    data: RDD[Array[LabelledPoint]],
    alphaInit: RDD[DenseVector[Double]],
    v: DenseVector[Double],
    round: Int
  ): Boolean
}

class DualityGapConvergenceChecker(epsilon: Double = 10E-8, maxRounds: Int = 1000, verbose: Boolean = true)
  extends ConvergenceChecker {

  def hasConverged(
    loss: Loss[_<:RealFunction, _<:RealFunction],
    regularizer: Regularizer,
    n: Long,
    data: RDD[Array[LabelledPoint]],
    alpha: RDD[DenseVector[Double]],
    v: DenseVector[Double],
    round: Int
  ) = {

    val primalObj = computePrimalObjective(data, loss, regularizer, n, regularizer.dualGradient(v))
    val dualObj = computeDualObjective(data, loss, regularizer, n, v, alpha)
    val gap = primalObj - dualObj

    if (verbose) {
      println(
        s"Iteration: $round",
        s"Objective Value: $primalObj",
        s"Dual Objective Value: $dualObj",
        s"Duality Gap: $gap"
      )
    }

    round > maxRounds || gap < epsilon
  }
}
