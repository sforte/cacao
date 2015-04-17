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
   * @param data Data points on which we wish to train the model
   * @param model The model we are training (logistic,svm,ridge etc.)
   * @param localSolver Method used to solve subproblems on local partitions
   * @param numRounds Number of outer iterations of CoCoA.
   * @param beta scaling parameter. beta=1 gives averaging, beta=K=data.partitions.size gives (aggressive) adding
   * @return The optimal (or near optimal) w and alpha vectors
   */
  def runCoCoA [ModelType<:PrimalDualModel] (
    sc: SparkContext,
    data: RDD[LabeledPoint],
    model: ModelType,
    localSolver: LocalSolverTrait[ModelType],
    numRounds: Int,
    beta: Double,
    seed: Int) : (DenseVector, RDD[DenseVector]) = {

    val n = data.count()
    val lambda = model.lambda

    // Initial feasible value for the alphas
    val alphaInit = data.map(pt => model.initAlpha(pt.label))

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
        partitionUpdate(model,n,_,localSolver,w,scaling,seed+t),preservesPartitioning=true).persist()

      alphaArr = updates.map(kv => kv._2)
      val primalUpdates = updates.map(kv => kv._1).reduce(plus)
      w = plus(times(primalUpdates,scaling),w)

      println(s"Iteration: $t")
      OptUtils.printSummaryStatsPrimalDual("CoCoA", dataArr, model, w, alphaArr)
    }

    println(Calendar.getInstance.getTime)

    OptUtils.printSummaryStatsPrimalDual("CoCoA", dataArr, model, w, alphaArr)

    (w, alphaArr)
  }

  /**
   * Performs one round of local updates using a given local dual algorithm, 
   * here locaSDCA. Will perform localIters many updates per worker.
   * @param zipData Partition data zipped with corresponding alphas
   * @param localSolver Method to be used locally to optimize on the partition
   * @param wInit Current w vector
   * @param scaling this is the scaling factor beta/K in the paper
   * @return
   */
  private def partitionUpdate [ModelType<:PrimalDualModel] (
    model: ModelType,
    n: Long,
    zipData: Iterator[(DenseVector, Array[LabeledPoint])],
    localSolver: LocalSolverTrait[ModelType],
    wInit: Vector,
    scaling: Double,
    seed: Int): Iterator[(DenseVector, DenseVector)] = {

    val zipPair = zipData.next()
    val localData = zipPair._2
    var alpha = zipPair._1
    val alphaOld = alpha.copy

    val (deltaAlpha, deltaW) = localSolver.optimize(model, n, localData, wInit.asInstanceOf[DenseVector], alpha, seed)

    alpha = plus(alphaOld,times(deltaAlpha,scaling))

    Iterator((deltaW, alpha))
  }
}
