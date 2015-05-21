package optimizers.distributed

import breeze.linalg.{Vector, DenseVector}
import models.loss.LogisticLoss
import models.{Regularizer, Loss, RealFunction}
import models.regularizer.L1Regularizer
import optimizers.coordinate.BrentMethodOptimizerWithFirstDerivative
import optimizers.local.{PasscodeWild, SDCAOptimizer}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import utils.OptUtils
import vectors.{LazyMappedVector, LabelledPoint}
import vectors.VectorOps._

import scala.math._

object AccCoCoAForL1 {
  def optimize (
    sc: SparkContext, n: Long, sigma: Double,
    _data: RDD[LabelledPoint]): (RDD[DenseVector[Double]], DenseVector[Double]) = {

    val data = _data.mapPartitions(x=>Iterator(x.toArray), preservesPartitioning = true)
    var alpha = _data.map(_ => 0.0)
      .mapPartitions(x=>Iterator(new DenseVector(x.toArray)), preservesPartitioning = true)

    var v = DenseVector.zeros[Double](100000)

    val loss = new LogisticLoss

    val scOptimizer = new BrentMethodOptimizerWithFirstDerivative(1000)

    val localSolver = new SDCAOptimizer(scOptimizer, 10)

    val cocoa = new CoCoA(sc, localSolver, 100, 1.0, 0)

    val acccocoa = new AccCoCoA(sc, cocoa, 0)

    val epsilon = 0.001
    var B = 1.0

    var regularizer : Regularizer = null

    while (epsilon >= 0) {

      val oldlambda = if (regularizer != null) regularizer.lambda else 1.0

      regularizer = new SmoothedL1Regularizer(sigma, epsilon, B)

      v *= oldlambda / regularizer.lambda

      val (_alpha, _v) = acccocoa.optimize(loss, regularizer, n, data, alpha, v, epsilon)
      alpha = _alpha
      v = _v

      val primal = OptUtils.computePrimalObjective(data, loss,
        new L1Regularizer(sigma), n, regularizer.dualGradient(v))

////      val Bnew = primal / sigma
////      epsilon = epsilon*(Bnew*Bnew)/(B*B)
////
////      if (math.abs(B - Bnew) < 0.001)
////        epsilon /= 2.0
//
//      B = min(Bnew,B)

      B = min(2.0*B, primal/sigma)

//      B = min(primal/sigma, B)
//      epsilon /= 2

      println(s"epsilon = $epsilon, B = $B")
      OptUtils.printSummaryStatsPrimalDual("asdf", data, loss, regularizer, n, v, alpha)
    }

    null
  }

  class L1Regularizer(val lambda: Double) extends Regularizer {
    def primal(w: Vector[Double]) = l1norm(w)
    override def dualGradient(w: Vector[Double]): Vector[Double] = ???
  }

  class SmoothedL1Regularizer(sigma: Double, epsilon: Double, B: Double) extends Regularizer {

    require(sigma > 0.0 && epsilon > 0.0)

    val lambda = epsilon / (B * B)
    val p = sigma / lambda

    def primal(w: Vector[Double]) = 1.0/2*l2norm(w) + p*l1norm(w)

    def dualGradient(v: Vector[Double]) = {
      new LazyMappedVector(v, {case(i,x) => if (abs(x) > p) x - p*signum(x) else 0.0})
    }

    override def toString = s"${0.5*lambda} |w|^2_2 + $sigma |w|_1"
  }
}
