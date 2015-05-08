package optimizers.distributed

import java.util.Calendar

import breeze.linalg.{DenseVector, Vector}
import models._
import optimizers.LocalOptimizer
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import utils.OptUtils
import utils.OptUtils._
import vectors.LabelledPoint

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
    data: RDD[Array[LabelledPoint]],
    alphaInit: RDD[DenseVector[Double]],
    model: ModelType,
    localSolver: LocalOptimizer[ModelType],
    numRounds: Int,
    beta: Double,
    seed: Int,
    epsilon: Double = 0.0) : (Vector[Double], RDD[DenseVector[Double]]) = {

    val n = model.n
    val parts = data.count()
    val lambda = model.regularizer.lambda
    val scaling = beta / parts

    var alpha = alphaInit

    val v = (data zip alpha).map(p => (p._1 zip p._2.data)
      .map { case (LabelledPoint(y,x), a) => x * (a/(lambda*n))}.reduce(_+_))
        .reduce(_+_).toDenseVector

    println("\nRunning CoCoA on "+n+" data examples, distributed over "+parts+" workers")
    println(Calendar.getInstance.getTime)

    var t = 1
    var gap = computePrimalObjective(data, model, model.regularizer.dualGradient(v))
              - computeDualObjective(data, model, v, alpha)

    while (t <= numRounds && gap > epsilon) {
      val vv = sc.broadcast(v)
      val updates = (alpha zip data).mapPartitions(
        partitionUpdate(model,_,localSolver,vv,seed+t),preservesPartitioning=true).cache()

      alpha = (alpha zip updates.map(_._2)).map {
        case (alphaOld, deltaAlpha) => alphaOld + deltaAlpha * scaling }.cache()

      v += updates.map(_._1).reduce(_+_) * scaling

      println(s"Iteration: $t")
      if (t % 1 == 0) {
        gap = printSummaryStatsPrimalDual("CoCoA", data, model, v, alpha)
        println(epsilon)
      }

      t += 1
    }

    println(Calendar.getInstance.getTime)

    printSummaryStatsPrimalDual("CoCoA", data, model, v, alpha)

    (v, alpha)
  }

  def runCoCoA [ModelType<:DualModel] (
    sc: SparkContext,
    data: RDD[LabelledPoint],
    model: ModelType,
    localSolver: LocalOptimizer[ModelType],
    numRounds: Int,
    beta: Double,
    seed: Int,
    epsilon: Double = 0.0) : (Vector[Double], RDD[Double]) = {

    val alpha = data.map(pt => 0.0)

    val partData = data.mapPartitions(x=>Iterator(x.toArray), preservesPartitioning = true).cache()
    val partAlphas = alpha.mapPartitions(x=>Iterator(new DenseVector(x.toArray)), preservesPartitioning = true)

    val (vNew, alphaNew) = runCoCoA(sc, partData, partAlphas, model, localSolver, numRounds, beta, seed, epsilon)

    (vNew, alphaNew.flatMap(_.toArray))
  }

  /**
   * Performs one round of local updates using a given local dual algorithm, 
   * here locaSDCA. Will perform localIters many updates per worker.
   * @param zipData Partition data zipped with corresponding alphas
   * @param localSolver Method to be used locally to optimize on the partition
   * @param vInit Current w vector
   * @return
   */
  private def partitionUpdate [ModelType<:DualModel] (
    model: ModelType,
    zipData: Iterator[(DenseVector[Double], Array[LabelledPoint])],
    localSolver: LocalOptimizer[ModelType],
    vInit: Broadcast[DenseVector[Double]],
    seed: Int): Iterator[(DenseVector[Double], DenseVector[Double])] = {

    val zipPair = zipData.next()
    val localData = zipPair._2
    val alpha = zipPair._1
    val v = vInit.value

    val (deltaAlpha,deltaV) = localSolver.optimize(model, localData, v, alpha)

    Iterator((deltaV, deltaAlpha))
  }
}
