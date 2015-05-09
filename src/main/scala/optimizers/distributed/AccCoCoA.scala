package optimizers.distributed

import breeze.linalg.{DenseVector, Vector}
import models._
import optimizers.LocalOptimizer
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import utils.OptUtils
import utils.OptUtils.{computeDualObjective, computePrimalObjective}
import vectors.VectorOps._
import vectors.{LabelledPoint, LazySumVector}

object AccCoCoA {
  /**
   * CoCoA - Communication-efficient distributed dual Coordinate Ascent.
   * 
   * @param sc Spark context
   * @param data Data points on which we wish to train the model
   * @param loss The model we are training (logistic,svm,ridge etc.)
   * @param localSolver Method used to solve subproblems on local partitions
   * @param numRounds Number of outer iterations of CoCoA.
   * @param beta scaling parameter. beta=1 gives averaging, beta=K=data.partitions.size gives (aggressive) adding
   * @return The optimal (or near optimal) w and alpha vectors
   */
  def runCoCoA [LossType<:Loss[RealFunction,RealFunction]] (
    sc: SparkContext,
    data: RDD[LabelledPoint],
    loss: LossType,
    regularizer: Regularizer,
    n: Long,
    localSolver: LocalOptimizer[LossType],
    numRounds: Int,
    beta: Double,
    seed: Int) : (Vector[Double], RDD[DenseVector[Double]]) = {

    val lambda = regularizer.lambda
    val R = math.sqrt(data.map(pt => l2norm(pt.features)).max)
    val gamma = 4.0
    val kappa = R*R/(gamma*n) - lambda

    if (kappa <= 0) {
      println(s"kappa = $kappa <= 0")
      return null
    }

    val mu = lambda / 2
    val rho = mu + kappa

    val eta = math.sqrt(mu/rho)
    val bbeta = (1-eta)/(1+eta)

    val alpha = data.map(pt => 0.0)

    val d = data.first().features.size

    var v : Vector[Double] = DenseVector.zeros[Double](d)
    var w : Vector[Double] = DenseVector.zeros[Double](d)
    var y = DenseVector.zeros[Double](d)

    val partData = data.mapPartitions(x=>Iterator(x.toArray), preservesPartitioning = true).cache()
    var partAlphas = alpha.mapPartitions(x=>Iterator(new DenseVector(x.toArray)), preservesPartitioning = true)

    var xi = (1 + 1/(eta*eta)) *
      (computePrimalObjective(partData, loss, regularizer, n, regularizer.dualGradient(v))
        - computeDualObjective(partData, loss, regularizer, n, v, partAlphas))

    var accreg = regularizer

    for (t <- 2 to 100) {

      val eps = xi * eta / (2*(1 + 1/(eta*eta)))

      accreg = new AccRegularizer(regularizer, kappa, y)

      y = (w * (-bbeta)).toDenseVector

      val asdf = CoCoA.runCoCoA(sc, partData, partAlphas, loss, regularizer, n,
                                localSolver, numRounds, beta, seed, eps)

      v = asdf._1
      w = accreg.dualGradient(asdf._1)
      partAlphas = asdf._2

      y = (w * (1 + bbeta) + y).toDenseVector

      xi *= 1 - eta/2

      OptUtils.printSummaryStatsPrimalDual("AccCoCoA", partData, loss,
        regularizer, n,
        (partData zip partAlphas).flatMap(p=> p._1 zip p._2.data)
          .map { case(LabelledPoint(y,x), a) => x * (a/(lambda*n)) }.reduce(_+_).toDenseVector,
        partAlphas)
    }

    null
  }
}

class AccRegularizer(reg: Regularizer, kappa: Double, y: DenseVector[Double]) extends Regularizer {

  def lambda = reg.lambda

  def primal(w: Vector[Double]) = reg.primal(w) - kappa*(w dot y)

//  def dualGradient(w: Vector[Double]) = reg.dualGradient(w + (y * kappa))
  def dualGradient(w: Vector[Double]) = w match {
    case (w: DenseVector[Double]) => reg.dualGradient(new LazySumVector(w, y, kappa))
  }
}