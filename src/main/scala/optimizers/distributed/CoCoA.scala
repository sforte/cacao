package optimizers.distributed

import breeze.linalg.DenseVector
import models.Loss
import optimizers.local.SDCAOptimizer
import optimizers.{LocalOptimizer, DistributedOptimizer}
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import utils.{DualityGapConvergenceChecker, ConvergenceChecker}
import vectors.LabelledPoint
import models.Model

class CoCoA[-LossType<:Loss[_,_]] (
  @transient sc: SparkContext,
  beta: Double = 1.0,
  localSolver: LocalOptimizer[LossType] = new SDCAOptimizer,
  convergenceChecker: ConvergenceChecker[LossType] = new DualityGapConvergenceChecker)
  extends DistributedOptimizer[LossType] {

  def optimize (
    model: Model[LossType],
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

    while (!convergenceChecker.hasConverged(model, data, alpha, v, t)) {

      val vv = sc.broadcast(v)

      val updates = (alpha zip data).mapPartitions(
        CoCoA.partitionUpdate(model, localSolver, _, vv),preservesPartitioning=true).cache()

      alpha = (alpha zip updates.map(_._2)).map ({
        case (alphaOld, deltaAlpha) => alphaOld + deltaAlpha * scaling }).cache()

      v += updates.map(_._1).reduce(_+_) * scaling

      t += 1
    }

    (alpha, v)
  }
}

object CoCoA {
  def partitionUpdate [LossType<:Loss[_,_]] (
    model: Model[LossType],
    localSolver: LocalOptimizer[LossType],
    zipData: Iterator[(DenseVector[Double], Array[LabelledPoint])],
    vInit: Broadcast[DenseVector[Double]]): Iterator[(DenseVector[Double], DenseVector[Double])] = {

    val zipPair = zipData.next()
    val localData = zipPair._2
    val alpha = zipPair._1
    val v = vInit.value

    val (deltaAlpha,deltaV) = localSolver.optimize(model, localData, v, alpha)

    Iterator((deltaV, deltaAlpha))
  }
}
