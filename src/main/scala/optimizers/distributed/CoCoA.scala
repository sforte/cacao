package optimizers.distributed

import java.util.Calendar
import breeze.linalg.DenseVector
import models.{Loss, Regularizer, RealFunction}
import optimizers.{LocalOptimizer, DistributedOptimizer}
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import utils.OptUtils._
import vectors.LabelledPoint

class CoCoA[-LossType<:Loss[RealFunction,RealFunction]]
  (@transient sc: SparkContext, localSolver: LocalOptimizer[LossType], numRounds: Int, beta: Double, seed: Int, asd: Double = 1.0)
  extends DistributedOptimizer[LossType] {

  def optimize(
    loss: LossType, regularizer: Regularizer, n: Long,
    data: RDD[Array[LabelledPoint]],
    alphaInit: RDD[DenseVector[Double]],
    vInit: DenseVector[Double],
    epsilon: Double
  ): (RDD[DenseVector[Double]], DenseVector[Double]) = {

    data.cache()
    val parts = data.count()
    val scaling = beta / parts

    var alpha = alphaInit
    alpha.cache()
    val v = vInit

    println("\nRunning CoCoA on "+n+" data examples, distributed over "+parts+" workers")
    println(Calendar.getInstance.getTime)

    var t = 1
    var gap = (computePrimalObjective(data, loss, regularizer, n, regularizer.dualGradient(v))
                 - computeDualObjective(data, loss, regularizer, n, v, alpha))

    while (t <= numRounds /*&& gap > epsilon*/) {
      val vv = sc.broadcast(v)

      val updates = (alpha zip data).mapPartitions(
        CoCoA.partitionUpdate(loss,regularizer,n,localSolver,_,vv,seed+t),preservesPartitioning=true).cache()

      alpha = (alpha zip updates.map(_._2)).map ({
        case (alphaOld, deltaAlpha) => alphaOld + deltaAlpha * scaling }).cache()

      v += updates.map(_._1).reduce(_+_) * scaling

      println(s"Iteration: $t")
      if (t % 1 == 0) {
        gap = printSummaryStatsPrimalDual("CoCoA", data, loss, regularizer, n, v, alpha, asd)
        println(epsilon)
      }

      t += 1

//      if (t % 100 == 0) alpha.checkpoint()
    }

    println(Calendar.getInstance.getTime)

    printSummaryStatsPrimalDual("CoCoA", data, loss, regularizer, n, v, alpha, asd)

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
    vInit: Broadcast[DenseVector[Double]],
    seed: Int): Iterator[(DenseVector[Double], DenseVector[Double])] = {

    val zipPair = zipData.next()
    val localData = zipPair._2
    val alpha = zipPair._1
    val v = vInit.value

    val (deltaAlpha,deltaV) = localSolver.optimize(loss, regularizer, n, localData, v, alpha)

    Iterator((deltaV, deltaAlpha))
  }
}
