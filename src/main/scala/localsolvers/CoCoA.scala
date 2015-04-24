package localsolvers

import java.util.Calendar
import distopt.utils.OptUtils
import distopt.utils.VectorOps._
import models.DualModel
import org.apache.spark.mllib.linalg.{DenseVector, Vector}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

class CoCoA[-ModelType<:DualModel] (localSolver: LocalOptimizer[ModelType], numRounds: Int, beta: Double)
    extends PrimalDualSolver[ModelType, RDD[LabeledPoint], RDD[Double], Vector] {

  override def optimize(model: ModelType, data: RDD[LabeledPoint], alphaInit: RDD[Double], w: Vector): (RDD[Double], Vector) = {

    val n = data.count()
    val lambda = model.lambda

    // Initial feasible value for the alphas
//    val alphaInit = data.map(pt => model.initAlpha(pt.label))

    val parts = data.partitions.size

    // We group partitions data in an Array and we zip it with an array of alphas
    val zipData: RDD[(DenseVector, Array[LabeledPoint])] =
      (alphaInit zip data).mapPartitions(x => Iterator(x.toArray), preservesPartitioning = true)
        .map(x => (new DenseVector(x.map(_._1).toArray), x.map(_._2).toArray))

    var alphaArr = zipData.map(_._1).cache()
    val dataArr = zipData.map(_._2).cache()

    val scaling = beta / parts

    // computing the initial w vector, given the initial feasible alphas
    var w = new DenseVector(
      (alphaInit zip data)
        .map { case (a, LabeledPoint(_,x)) => times(x,a/(lambda*n)) }
        .reduce(plus).toArray)

    println("\nRunning CoCoA on "+n+" data examples, distributed over "+parts+" workers")
    println(Calendar.getInstance.getTime)

    for(t <- 1 to numRounds) {

      val zipData = alphaArr zip dataArr

      val updates = zipData.mapPartitions(
        partitionUpdate(model,n,_,localSolver,w,scaling),preservesPartitioning=true).persist()

      alphaArr = updates.map(kv => kv._2)
      val primalUpdates = updates.map(kv => kv._1).reduce(plus)
      w = plus(times(primalUpdates,scaling),w)

      println(s"Iteration: $t")
      OptUtils.printSummaryStatsPrimalDual("CoCoA", dataArr, model, w, alphaArr)
    }

    println(Calendar.getInstance.getTime)

    OptUtils.printSummaryStatsPrimalDual("CoCoA", dataArr, model, w, alphaArr)

    (null, w)
  }

  private def partitionUpdate [ModelType<:DualModel] (
    model: ModelType,
    n: Long,
    zipData: Iterator[(DenseVector, Array[LabeledPoint])],
    localSolver: LocalOptimizer[ModelType],
    wInit: Vector,
    scaling: Double): Iterator[(DenseVector, DenseVector)] = {

    val zipPair = zipData.next()
    val localData = zipPair._2
    var alpha = zipPair._1
    val alphaOld = alpha.copy

    val (deltaAlpha,deltaW) = localSolver.optimize(model, localData, wInit.asInstanceOf[DenseVector], alpha)

    alpha = plus(alphaOld,times(deltaAlpha,scaling))

    Iterator((deltaW, alpha))
  }
}
