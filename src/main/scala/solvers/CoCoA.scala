package distopt.solvers

import java.util.Calendar

import distopt.utils.OptUtils._
import distopt.utils.VectorOps._
import distopt.utils._
import localsolvers._
import models._
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.{Vectors, SparseVector, DenseVector, Vector}
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
  def runCoCoA [ModelType<:DualModel] (
    sc: SparkContext,
    data: RDD[Array[LabeledPoint]],
    alpha: RDD[DenseVector],
    model: ModelType,
    localSolver: LocalOptimizer[ModelType],
    numRounds: Int,
    beta: Double,
    seed: Int,
    epsilon: Double = 0.0) : (Vector, RDD[DenseVector]) = {

    val parts = data.count()
    val lambda = model.regularizer.lambda

    val n = model.n
    val d = data.first()(0).features.size

    val zipData = alpha zip data
    var alphaArr = zipData.map(_._1).cache()
    val dataArr = zipData.map(_._2).cache()

    val scaling = beta / parts

    var v = plus(Vectors.zeros(d), (dataArr zip alphaArr).flatMap(p=> (p._1 zip p._2.values))
      .map { case(LabeledPoint(y,x), a) => times(x,a/(lambda*n)) }
        .reduce(plus))

    println("\nRunning CoCoA on "+n+" data examples, distributed over "+parts+" workers")
    println(Calendar.getInstance.getTime)

    var t = 1
    var gap = computePrimalObjective(data, model, model.regularizer.dualGradient(v))
              - computeDualObjective(data, model, v, alphaArr)

    while (t <= numRounds && gap > epsilon) {

      val zipData = alphaArr zip dataArr

      val updates = zipData.mapPartitions(
        partitionUpdate(model,_,localSolver,v,scaling,seed+t),preservesPartitioning=true).persist()

      alphaArr = updates.map(kv => kv._2)
      val primalUpdates = updates.map(kv => kv._1).reduce(plus)
      v = plus(times(primalUpdates,scaling),v)

      println(s"Iteration: $t")
      gap = printSummaryStatsPrimalDual("CoCoA", dataArr, model, v, alphaArr)

      val objVal = computePrimalObjective(dataArr, model, model.regularizer.dualGradient(v))
      val dualObjVal = computeDualObjective(dataArr, model, v, alphaArr)
      gap = objVal - dualObjVal

//      gap = computePrimalObjective(dataArr, model, model.regularizer.dualGradient(v))
//            - computeDualObjective(dataArr, model, v, alphaArr)

      println(gap, epsilon)
      t += 1
    }

    println(Calendar.getInstance.getTime)

//    OptUtils.printSummaryStatsPrimalDual("CoCoA", dataArr, model, v, alphaArr)

    (v, alphaArr)
  }

  def runCoCoA [ModelType<:DualModel] (
    sc: SparkContext,
    data: RDD[LabeledPoint],
    model: ModelType,
    localSolver: LocalOptimizer[ModelType],
    numRounds: Int,
    beta: Double,
    seed: Int,
    epsilon: Double = 0.0) : (Vector, RDD[Double]) = {

    val alpha = data.map(pt => 0.0)
    val v = Vectors.zeros(data.first().features.size)

    val lambda = model.regularizer.lambda
    val n = model.n

    val partData = data.mapPartitions(x=>Iterator(x.toArray), preservesPartitioning = true)
    val partAlphas = alpha.mapPartitions(x=>Iterator(new DenseVector(x.toArray)), preservesPartitioning = true)

    val (vNew,alphaNew) = runCoCoA(sc, partData, partAlphas, model, localSolver, numRounds, beta, seed, epsilon)

    (vNew, alphaNew.flatMap(_.values))
  }

  /**
   * Performs one round of local updates using a given local dual algorithm, 
   * here locaSDCA. Will perform localIters many updates per worker.
   * @param zipData Partition data zipped with corresponding alphas
   * @param localSolver Method to be used locally to optimize on the partition
   * @param vInit Current w vector
   * @param scaling this is the scaling factor beta/K in the paper
   * @return
   */
  private def partitionUpdate [ModelType<:DualModel] (
    model: ModelType,
    zipData: Iterator[(DenseVector, Array[LabeledPoint])],
    localSolver: LocalOptimizer[ModelType],
    vInit: Vector,
    scaling: Double,
    seed: Int): Iterator[(DenseVector, DenseVector)] = {

    val zipPair = zipData.next()
    val localData = zipPair._2
    var alpha = zipPair._1
    val alphaOld = alpha.copy

    val (deltaAlpha,deltaV) = localSolver.optimize(model, localData, vInit.asInstanceOf[DenseVector], alpha)

    alpha = plus(alphaOld,times(deltaAlpha,scaling))

    Iterator((deltaV, alpha))
  }
}
