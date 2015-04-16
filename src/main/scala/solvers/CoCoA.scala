package distopt.solvers

import java.util.Calendar

import distopt.utils.VectorOps._
import distopt.utils._
import localsolvers.LocalSolverTrait
import models.PrimalDualModel
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.{DenseVector, Vector}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

object CoCoA {
  /**
   * CoCoA - Communication-efficient distributed dual Coordinate Ascent.
   * 
   * @param sc Spark context
   * @param localSolver Method used to solve subproblems on local partitions
   * @param numRounds number of outer iterations T in the paper
   * @param beta scaling parameter. beta=1 gives averaging, beta=K=data.partitions.size gives (aggressive) adding
   * @return
   */
  def runCoCoA (
    sc: SparkContext,
    data: RDD[LabeledPoint],
    problem: PrimalDualModel,
    localSolver: LocalSolverTrait,
    numRounds: Int,
    beta: Double,
    seed: Int) : (DenseVector, RDD[DenseVector]) = {

    val n = data.count()
    val primalLoss = problem.primalLoss
    val dualLoss = problem.dualLoss
    val lambda = problem.lambda
    val alphaInit = data.map(pt => problem.initAlpha(pt.label))

    val parts = data.partitions.size

    val zipData: RDD[(DenseVector, Array[LabeledPoint])] =
      (alphaInit zip data).mapPartitions(x => Iterator(x.toArray), preservesPartitioning = true)
      .map(x => (new DenseVector(x.map(_._1).toArray), x.map(_._2).toArray))

    zipData.mapPartitions(x => Iterator(x.next()._2.size)).foreach(println)

    var alphaArr = zipData.map(_._1).cache()
    val dataArr = zipData.map(_._2).cache()

    val scaling = beta / parts

    var w = new DenseVector(
      (alphaInit zip data)
        .map { case (a, LabeledPoint(_,x)) => times(x,a/(lambda*n)) }
        .reduce(plus).toArray)

    println("\nRunning CoCoA on "+n+" data examples, distributed over "+parts+" workers")
    println(Calendar.getInstance.getTime)

    for(t <- 1 to numRounds) {

      val zipData = alphaArr zip dataArr

      val updates = zipData.mapPartitions(
        partitionUpdate(_,localSolver,w,scaling,seed+t),preservesPartitioning=true).persist()

      alphaArr = updates.map(kv => kv._2)
      val primalUpdates = updates.map(kv => kv._1).reduce(plus)
      w = plus(times(primalUpdates,scaling),w)

      println(s"Iteration: $t")
      if (t % 10 == 0)
      OptUtils.printSummaryStatsPrimalDual("CoCoA", dataArr, problem, w, alphaArr)
    }

    println(Calendar.getInstance.getTime)

    OptUtils.printSummaryStatsPrimalDual("CoCoA", dataArr, problem, w, alphaArr)

    (w, alphaArr)
  }

  /**
   * Performs one round of local updates using a given local dual algorithm, 
   * here locaSDCA. Will perform localIters many updates per worker.
   * @param zipData ...
   * @param wInit ...
   * @param scaling this is the scaling factor beta/K in the paper
   * @param seed ...
   * @return
   */
  private def partitionUpdate(
    zipData: Iterator[(DenseVector, Array[LabeledPoint])],
    localSolver: LocalSolverTrait,
    wInit: Vector,
    scaling: Double,
    seed: Int): Iterator[(DenseVector, DenseVector)] = {

    val zipPair = zipData.next()
    val localData = zipPair._2
    var alpha = zipPair._1
    val alphaOld = alpha.copy

    val (deltaAlpha, deltaW) = localSolver.optimize(localData, wInit.asInstanceOf[DenseVector], alpha, seed)

    alpha = plus(alphaOld,times(deltaAlpha,scaling))

    Iterator((deltaW, alpha))
  }
}
