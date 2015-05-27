package optimizers.distributed

import breeze.linalg.DenseVector
import models.{Loss, Regularizer, RealFunction}
import optimizers.{LocalOptimizer, DistributedOptimizer}
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import utils.{DualityGapConvergenceChecker, ConvergenceChecker}
import vectors.LabelledPoint

class CoCoA[-LossType<:Loss[RealFunction,RealFunction]]
  (@transient sc: SparkContext, localSolver: LocalOptimizer[LossType],
   beta: Double = 1.0, convergenceChecker: ConvergenceChecker = new DualityGapConvergenceChecker)
  extends DistributedOptimizer[LossType] {

  def optimize (
    loss: LossType, regularizer: Regularizer, n: Long,
    data: RDD[Array[LabelledPoint]],
    alphaInit: RDD[DenseVector[Double]],
    vInit: DenseVector[Double]
  ): (RDD[DenseVector[Double]], DenseVector[Double]) = {

    data.cache()
    val parts = data.count()
    val scaling = beta / parts

    var alpha = alphaInit
    alpha.cache()
    val v = vInit

    var t = 1

    while (!convergenceChecker.hasConverged(loss, regularizer, n, data, alpha, v, t)) {

      val vv = sc.broadcast(v)

      val updates = (alpha zip data).mapPartitions(
        CoCoA.partitionUpdate(loss,regularizer,n,localSolver,_,vv),preservesPartitioning=true).cache()

      alpha = (alpha zip updates.map(_._2)).map ({
        case (alphaOld, deltaAlpha) => alphaOld + deltaAlpha * scaling }).cache()

      v += updates.map(_._1).reduce(_+_) * scaling

      t += 1
    }

    (alpha, v)
  }
}

object CoCoA {
  def partitionUpdate [LossType<:Loss[RealFunction,RealFunction]] (
    loss: LossType,
    regularizer: Regularizer,
    n: Long,
    localSolver: LocalOptimizer[LossType],
    zipData: Iterator[(DenseVector[Double], Array[LabelledPoint])],
    vInit: Broadcast[DenseVector[Double]]): Iterator[(DenseVector[Double], DenseVector[Double])] = {

    val zipPair = zipData.next()
    val localData = zipPair._2
    val alpha = zipPair._1
    val v = vInit.value

    val (deltaAlpha,deltaV) = localSolver.optimize(loss, regularizer, n, localData, v, alpha)

    Iterator((deltaV, deltaAlpha))
  }
}
